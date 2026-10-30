"""
Hard Communication Budget Manager for Federated Learning.
Measures exact network payload bytes, tracks cumulative usage,
and enforces hard communication budget constraints.
"""

from typing import Dict, Any, Optional
import os


class CommunicationBudgetManager:
    """
    Manages and strictly enforces a hard communication byte budget.
    """
    def __init__(
        self,
        total_budget_bytes: int,
        protocol_overhead_bytes: int = 128
    ):
        self.total_budget_bytes = total_budget_bytes
        self.protocol_overhead_bytes = protocol_overhead_bytes

        self.used_download_bytes = 0
        self.used_upload_bytes = 0
        self.total_used_bytes = 0
        self.deferred_transmissions_count = 0
        self.successful_transmissions_count = 0

    def get_remaining_budget(self) -> int:
        """Returns remaining available communication budget in bytes."""
        return max(0, self.total_budget_bytes - self.total_used_bytes)

    def is_exhausted(self) -> bool:
        """Returns True if the budget is completely exhausted."""
        return self.total_used_bytes >= self.total_budget_bytes

    def can_accommodate(self, estimated_bytes: int) -> bool:
        """Checks whether the requested bytes can fit within the remaining budget."""
        effective_bytes = estimated_bytes + self.protocol_overhead_bytes
        return (self.total_used_bytes + effective_bytes) <= self.total_budget_bytes

    def request_transmission(
        self,
        download_bytes: int,
        upload_bytes: int
    ) -> bool:
        """
        Attempts to allocate communication capacity for a client exchange.
        Returns True if approved, False if deferred due to budget exhaustion.
        """
        required_bytes = download_bytes + upload_bytes + (2 * self.protocol_overhead_bytes)
        
        if (self.total_used_bytes + required_bytes) <= self.total_budget_bytes:
            # Approve and deduct
            self.used_download_bytes += download_bytes + self.protocol_overhead_bytes
            self.used_upload_bytes += upload_bytes + self.protocol_overhead_bytes
            self.total_used_bytes += required_bytes
            self.successful_transmissions_count += 1
            return True
        else:
            # Reject and log deferral
            self.deferred_transmissions_count += 1
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Returns a complete budget accounting dictionary."""
        return {
            "total_budget_bytes": self.total_budget_bytes,
            "total_budget_mb": self.total_budget_bytes / (1024.0 ** 2),
            "used_download_bytes": self.used_download_bytes,
            "used_upload_bytes": self.used_upload_bytes,
            "total_used_bytes": self.total_used_bytes,
            "total_used_mb": self.total_used_bytes / (1024.0 ** 2),
            "remaining_budget_bytes": self.get_remaining_budget(),
            "remaining_budget_mb": self.get_remaining_budget() / (1024.0 ** 2),
            "utilization_percent": (self.total_used_bytes / self.total_budget_bytes) * 100.0 if self.total_budget_bytes > 0 else 0.0,
            "is_exhausted": self.is_exhausted(),
            "successful_transmissions": self.successful_transmissions_count,
            "deferred_transmissions": self.deferred_transmissions_count
        }
