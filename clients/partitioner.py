"""
Client Partitioner Module for Federated Traffic Learning.
Implements reproducible IID and Non-IID sensor partitioning across 5 virtual clients.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


class ClientPartition:
    """
    Represents a single virtual client's data partition metadata.
    """
    def __init__(
        self,
        client_id: int,
        sensor_indices: np.ndarray,
        mean_flow: float,
        std_flow: float,
        num_sensors: int,
        local_adj: np.ndarray,
        local_laplacian: np.ndarray
    ):
        self.client_id = client_id
        self.sensor_indices = sensor_indices
        self.mean_flow = mean_flow
        self.std_flow = std_flow
        self.num_sensors = num_sensors
        self.local_adj = local_adj
        self.local_laplacian = local_laplacian

    def __repr__(self) -> str:
        return (f"Client {self.client_id}: {self.num_sensors} sensors, "
                f"Mean Flow: {self.mean_flow:.2f}, Std Flow: {self.std_flow:.2f}")


class FederatedPartitioner:
    """
    Partitions PEMS04 307 sensors into 5 virtual clients under
    IID (stratified balanced) or Non-IID (traffic-volume tier) configurations.
    """
    def __init__(
        self,
        num_clients: int = 5,
        num_nodes: int = 307,
        random_seed: int = 42
    ):
        self.num_clients = num_clients
        self.num_nodes = num_nodes
        self.random_seed = random_seed

    def partition_iid(
        self,
        raw_train_data: np.ndarray,
        global_adj: np.ndarray,
        global_laplacian: np.ndarray
    ) -> List[ClientPartition]:
        """
        Creates an IID partition using stratified sampling on sensor mean traffic flows.
        Each client receives ~61-62 sensors with statistically indistinguishable flow distributions.
        """
        np.random.seed(self.random_seed)
        mean_flow_per_sensor = raw_train_data[:, :, 0].mean(axis=0)  # shape: (307,)
        
        # Sort sensors by mean flow and distribute cyclically across clients (stratified)
        sorted_sensors = np.argsort(mean_flow_per_sensor)
        client_sensor_lists: List[List[int]] = [[] for _ in range(self.num_clients)]
        
        for rank, s_idx in enumerate(sorted_sensors):
            # Alternating serpentine allocation for maximum distributional balance
            c = rank % self.num_clients if (rank // self.num_clients) % 2 == 0 else (self.num_clients - 1 - (rank % self.num_clients))
            client_sensor_lists[c].append(int(s_idx))

        partitions = []
        for c_id, s_list in enumerate(client_sensor_lists):
            s_arr = np.array(sorted(s_list), dtype=int)
            local_flows = mean_flow_per_sensor[s_arr]
            local_adj = global_adj[np.ix_(s_arr, s_arr)]
            local_lap = global_laplacian[np.ix_(s_arr, s_arr)]
            
            p = ClientPartition(
                client_id=c_id,
                sensor_indices=s_arr,
                mean_flow=float(local_flows.mean()),
                std_flow=float(local_flows.std()),
                num_sensors=len(s_arr),
                local_adj=local_adj,
                local_laplacian=local_lap
            )
            partitions.append(p)

        return partitions

    def partition_non_iid(
        self,
        raw_train_data: np.ndarray,
        global_adj: np.ndarray,
        global_laplacian: np.ndarray
    ) -> List[ClientPartition]:
        """
        Creates a Non-IID partition using volume-tier clustering.
        Sensors are sorted into 5 distinct traffic-volume regimes:
        Client 0: Low-volume feeder routes
        Client 1: Moderate-low residential/secondary routes
        Client 2: Intermediate urban arterials
        Client 3: Heavy commercial corridors
        Client 4: High-volume highway/freeway bottlenecks
        """
        mean_flow_per_sensor = raw_train_data[:, :, 0].mean(axis=0)  # shape: (307,)
        sorted_sensors = np.argsort(mean_flow_per_sensor)
        
        # Split into 5 contiguous quintiles
        sensor_groups = np.array_split(sorted_sensors, self.num_clients)

        partitions = []
        for c_id, s_arr in enumerate(sensor_groups):
            s_arr = np.array(sorted(s_arr), dtype=int)
            local_flows = mean_flow_per_sensor[s_arr]
            local_adj = global_adj[np.ix_(s_arr, s_arr)]
            local_lap = global_laplacian[np.ix_(s_arr, s_arr)]

            p = ClientPartition(
                client_id=c_id,
                sensor_indices=s_arr,
                mean_flow=float(local_flows.mean()),
                std_flow=float(local_flows.std()),
                num_sensors=len(s_arr),
                local_adj=local_adj,
                local_laplacian=local_lap
            )
            partitions.append(p)

        return partitions
