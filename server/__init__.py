"""
Server Package for Federated Traffic Learning.
"""

from server.fedavg import aggregate_fedavg, validate_client_update
from server.coordinator import FederatedCoordinator
from server.proposed_aggregation import aggregate_befa, compute_befa_weights
from server.befa_coordinator import BEFACoordinator

__all__ = [
    "aggregate_fedavg",
    "validate_client_update",
    "FederatedCoordinator",
    "aggregate_befa",
    "compute_befa_weights",
    "BEFACoordinator"
]
