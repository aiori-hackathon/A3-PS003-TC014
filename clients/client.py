"""
Virtual Federated Client Implementation for Traffic Prediction.
Manages local dataset partition, local STGCN model, local training loop,
and communication byte measurement.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stgcn import CompactSTGCN
from models.graph_layers import compute_chebyshev_polynomials
from data.dataset_loader import TrafficDataset
from communication.serialization import measure_payload_bytes


class VirtualClient:
    """
    Simulated traffic prediction client (e.g. urban sub-district or intersection group).
    """
    def __init__(
        self,
        client_id: int,
        sensor_indices: np.ndarray,
        train_x: np.ndarray,
        train_y: np.ndarray,
        local_laplacian: np.ndarray,
        model_config: Dict[str, int],
        batch_size: int = 64,
        device: Optional[torch.device] = None
    ):
        self.client_id = client_id
        self.sensor_indices = sensor_indices
        self.num_sensors = len(sensor_indices)
        self.num_samples = len(train_x)
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size

        # Build local DataLoader
        self.dataset = TrafficDataset(train_x, train_y)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Precompute local Chebyshev polynomials: shape (K, N_local, N_local)
        k_order = model_config.get("k_order", 2)
        cheb_poly_np = compute_chebyshev_polynomials(local_laplacian, k_order=k_order)
        self.cheb_poly = torch.from_numpy(np.stack(cheb_poly_np, axis=0)).float().to(self.device)

        # Initialize local model with identical architecture
        self.model = CompactSTGCN(
            num_features=model_config.get("num_features", 3),
            in_steps=model_config.get("in_steps", 12),
            out_steps=model_config.get("out_steps", 3),
            hidden_channels=model_config.get("hidden_channels", 32),
            k_order=k_order,
            kt=model_config.get("kt", 3),
            dropout=model_config.get("dropout", 0.1)
        ).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer: Optional[optim.Optimizer] = None

    def receive_global_model(self, global_state_dict: Dict[str, torch.Tensor]) -> int:
        """
        Receives and loads global model parameters.
        Returns measured download bytes.
        """
        self.model.load_state_dict(global_state_dict)
        download_bytes = measure_payload_bytes(global_state_dict)
        return download_bytes

    def train_local(
        self,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> float:
        """
        Executes local model training on local partition data.
        Returns average training loss.
        """
        self.model.train()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        total_loss = 0.0
        num_batches = len(self.loader)

        for epoch in range(local_epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(x, self.cheb_poly)
                loss = self.criterion(preds, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (num_batches * local_epochs)

    def get_model_update(self) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        Exports the updated local model parameters, sample count, and measured upload bytes.
        Returns:
            (local_state_dict, num_samples, upload_bytes)
        """
        local_state = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
        upload_bytes = measure_payload_bytes(local_state)
        return local_state, self.num_samples, upload_bytes

    def get_recent_traffic_deviation(self, lookback_samples: int = 50) -> float:
        """
        Computes local traffic change score for event-awareness (used in BEFA):
        Measures variance/change in traffic flow over recent observations.
        """
        recent_x = self.dataset.x[-lookback_samples:, :, :, 0].numpy()
        recent_diff = np.abs(np.diff(recent_x, axis=1)).mean()
        return float(recent_diff)
