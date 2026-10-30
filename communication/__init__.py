"""
Communication Package for Federated Learning on PEMS04.
"""

from communication.serialization import serialize_model_state, deserialize_model_state, measure_payload_bytes
from communication.budget_manager import CommunicationBudgetManager
from communication.event_scheduler import EventScheduler
from communication.compression import compress_state_dict, decompress_state_dict, evaluate_compression_quality

__all__ = [
    "serialize_model_state",
    "deserialize_model_state",
    "measure_payload_bytes",
    "CommunicationBudgetManager",
    "EventScheduler",
    "compress_state_dict",
    "decompress_state_dict",
    "evaluate_compression_quality"
]
