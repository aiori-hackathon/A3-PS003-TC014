"""
Dataset Loader for PEMS04 Traffic Flow Forecasting.
Ensures strict chronological splitting (70% train, 10% val, 20% test)
and zero data leakage by computing normalization statistics exclusively
on the training partition.
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for spatial-temporal traffic forecasting windows.
    X shape: (num_samples, in_steps, num_nodes, num_features)
    Y shape: (num_samples, out_steps, num_nodes) [target: traffic flow]
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_raw_pems04(npz_path: str = "pems04.npz") -> np.ndarray:
    """
    Loads raw PEMS04 dataset from npz file.
    Expected shape: (16992, 307, 3)
    Features: [0: flow, 1: occupancy, 2: speed]
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"PEMS04 archive not found at: {npz_path}")
    
    with np.load(npz_path) as archive:
        if 'data' not in archive:
            raise KeyError(f"'data' key not found in archive. Found: {archive.files}")
        data = archive['data'].astype(np.float32)
        
    return data


def build_adjacency_matrix(
    csv_path: str = "data/adjacency/PEMS04.csv",
    num_nodes: int = 307,
    threshold: float = 0.1,
    epsilon: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs normalized adjacency matrix and normalized graph Laplacian
    from PEMS04 pairwise distance CSV using a thresholded Gaussian kernel.

    Returns:
        adj_mx: Symmetric adjacency matrix with self-loops in [0, 1]
        laplacian: Symmetric normalized graph Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Adjacency CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    dist_mx = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)

    for _, row in df.iterrows():
        u, v, cost = int(row['from']), int(row['to']), float(row['cost'])
        dist_mx[u, v] = cost

    # Valid distances to compute Gaussian kernel bandwidth
    valid_dists = dist_mx[~np.isinf(dist_mx)]
    sigma = float(valid_dists.std())

    # Gaussian kernel
    adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    mask = ~np.isinf(dist_mx)
    adj_mx[mask] = np.exp(- (dist_mx[mask] / sigma) ** 2)

    # Thresholding for sparsity
    adj_mx[adj_mx < threshold] = 0.0

    # Symmetrize road graph
    adj_sym = np.maximum(adj_mx, adj_mx.T)

    # Add self-loops
    np.fill_diagonal(adj_sym, 1.0)

    # Compute symmetric normalized Laplacian
    # D_inv_sqrt * A * D_inv_sqrt
    deg = np.sum(adj_sym, axis=1)
    deg_inv_sqrt = np.power(deg + epsilon, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    
    # Normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
    a_norm = d_mat @ adj_sym @ d_mat

    # Symmetric Laplacian: L_sym = I - A_norm
    laplacian = np.eye(num_nodes, dtype=np.float32) - a_norm

    return adj_sym, laplacian


def split_chronological(
    data: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sequential traffic data strictly chronologically.
    Total length: N
    Train: [0 : N_train]
    Val:   [N_train : N_train + N_val]
    Test:  [N_train + N_val : N]
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train: n_train + n_val]
    test_data = data[n_train + n_val:]

    return train_data, val_data, test_data


class TrafficDataScaler:
    """
    Standard Z-score scaler fit exclusively on training data.
    Preserves means and stds for traffic flow (channel 0) to invert
    model predictions back to original physical units (vehicles / 5-min).
    """
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, train_data: np.ndarray) -> 'TrafficDataScaler':
        # data shape: (T, N, F) -> compute mean and std across time and nodes per feature
        # shape: (1, 1, F)
        self.mean = np.mean(train_data, axis=(0, 1), keepdims=True)
        self.std = np.std(train_data, axis=(0, 1), keepdims=True)
        # Avoid division by zero
        self.std[self.std < 1e-5] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit before calling transform.")
        return (data - self.mean) / self.std

    def inverse_transform_flow(self, norm_flow: np.ndarray) -> np.ndarray:
        """
        Inverts normalized flow predictions (channel 0) to original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit before calling inverse_transform.")
        flow_mean = self.mean[0, 0, 0]
        flow_std = self.std[0, 0, 0]
        return norm_flow * flow_std + flow_mean

    def get_stats(self) -> Dict[str, float]:
        return {
            'flow_mean': float(self.mean[0, 0, 0]),
            'flow_std': float(self.std[0, 0, 0]),
            'occ_mean': float(self.mean[0, 0, 1]),
            'occ_std': float(self.std[0, 0, 1]),
            'speed_mean': float(self.mean[0, 0, 2]),
            'speed_std': float(self.std[0, 0, 2])
        }


def generate_sliding_windows(
    data: np.ndarray,
    in_steps: int = 12,
    out_steps: int = 3,
    target_channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates temporal sliding windows for forecasting.
    Input: data of shape (T, num_nodes, num_features)
    Output:
      x: shape (samples, in_steps, num_nodes, num_features)
      y: shape (samples, out_steps, num_nodes)
    """
    t, num_nodes, num_features = data.shape
    total_window = in_steps + out_steps
    num_samples = t - total_window + 1
    
    if num_samples <= 0:
        raise ValueError(f"Data length {t} is too short for window {total_window}")

    x = np.empty((num_samples, in_steps, num_nodes, num_features), dtype=np.float32)
    y = np.empty((num_samples, out_steps, num_nodes), dtype=np.float32)

    for i in range(num_samples):
        x[i] = data[i : i + in_steps]
        y[i] = data[i + in_steps : i + total_window, :, target_channel]

    return x, y


class PEMS04DataManager:
    """
    High-level manager for PEMS04 pipeline:
    1. Loads raw npz traffic matrix
    2. Builds normalized adjacency matrix and graph Laplacian
    3. Splits chronologically into Train (70%), Val (10%), Test (20%)
    4. Computes scaling statistics strictly on Train set
    5. Produces sliding window PyTorch DataLoaders
    """
    def __init__(
        self,
        npz_path: str = "pems04.npz",
        csv_path: str = "data/adjacency/PEMS04.csv",
        in_steps: int = 12,
        out_steps: int = 3,
        batch_size: int = 32
    ):
        self.npz_path = npz_path
        self.csv_path = csv_path
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.batch_size = batch_size

        # Data placeholders
        self.raw_data: Optional[np.ndarray] = None
        self.adj_matrix: Optional[np.ndarray] = None
        self.laplacian: Optional[np.ndarray] = None
        self.scaler = TrafficDataScaler()

        # Splits
        self.train_data_raw: Optional[np.ndarray] = None
        self.val_data_raw: Optional[np.ndarray] = None
        self.test_data_raw: Optional[np.ndarray] = None

        self.train_dataset: Optional[TrafficDataset] = None
        self.val_dataset: Optional[TrafficDataset] = None
        self.test_dataset: Optional[TrafficDataset] = None

    def prepare(self) -> 'PEMS04DataManager':
        # 1. Load raw data and adjacency
        self.raw_data = load_raw_pems04(self.npz_path)
        self.adj_matrix, self.laplacian = build_adjacency_matrix(self.csv_path, num_nodes=self.raw_data.shape[1])

        # 2. Split chronologically
        self.train_data_raw, self.val_data_raw, self.test_data_raw = split_chronological(self.raw_data)

        # 3. Fit scaler ONLY on train set
        self.scaler.fit(self.train_data_raw)

        # 4. Transform all sets using train scaler
        train_norm = self.scaler.transform(self.train_data_raw)
        val_norm = self.scaler.transform(self.val_data_raw)
        test_norm = self.scaler.transform(self.test_data_raw)

        # 5. Generate sliding windows
        x_train, y_train = generate_sliding_windows(train_norm, self.in_steps, self.out_steps)
        x_val, y_val = generate_sliding_windows(val_norm, self.in_steps, self.out_steps)
        x_test, y_test = generate_sliding_windows(test_norm, self.in_steps, self.out_steps)

        # 6. Wrap in PyTorch Datasets
        self.train_dataset = TrafficDataset(x_train, y_train)
        self.val_dataset = TrafficDataset(x_val, y_val)
        self.test_dataset = TrafficDataset(x_test, y_test)

        return self

    def get_dataloaders(self, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.train_dataset is None:
            raise RuntimeError("Manager must call .prepare() before accessing dataloaders.")
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
