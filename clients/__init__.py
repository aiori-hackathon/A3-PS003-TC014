"""
Clients Package for Federated Learning on PEMS04.
"""

from clients.client import VirtualClient
from clients.partitioner import FederatedPartitioner, ClientPartition

__all__ = ["VirtualClient", "FederatedPartitioner", "ClientPartition"]
