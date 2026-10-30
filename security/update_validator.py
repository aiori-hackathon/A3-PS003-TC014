"""
Lightweight Security and Update Validator Module.
Validates parameter integrity, enforces L2 norm clipping,
and rejects stale, duplicate, or non-finite client updates.
"""

import hashlib
from typing import Dict, Tuple, List, Optional, Any
import torch
from communication.serialization import serialize_model_state


class UpdateValidator:
    """
    Validates incoming federated model updates and enforces lightweight safeguards.
    """
    def __init__(
        self,
        max_norm_clip: float = 25.0,
        max_staleness_rounds: int = 2
    ):
        self.max_norm_clip = max_norm_clip
        self.max_staleness_rounds = max_staleness_rounds

        # Replay and duplicate detection
        self.seen_update_hashes: Dict[int, str] = {}
        self.rejection_log: List[Dict[str, Any]] = []

    def validate_and_clip(
        self,
        client_id: int,
        client_state: Dict[str, torch.Tensor],
        global_state: Dict[str, torch.Tensor],
        client_round: int,
        current_round: int
    ) -> Tuple[bool, Optional[Dict[str, torch.Tensor]], str]:
        """
        Validates update integrity and applies L2 norm clipping.
        Returns:
            (is_accepted, processed_state_dict_or_None, reason)
        """
        # 1. Staleness check
        staleness = current_round - client_round
        if staleness > self.max_staleness_rounds:
            reason = f"Stale update rejected (staleness={staleness} > max={self.max_staleness_rounds})"
            self._log_rejection(client_id, current_round, reason)
            return False, None, reason

        # 2. Key and shape compatibility check
        if set(client_state.keys()) != set(global_state.keys()):
            reason = "State dict keys mismatch with global model"
            self._log_rejection(client_id, current_round, reason)
            return False, None, reason

        for k, v in client_state.items():
            if v.shape != global_state[k].shape:
                reason = f"Tensor shape mismatch on '{k}': expected {global_state[k].shape}, got {v.shape}"
                self._log_rejection(client_id, current_round, reason)
                return False, None, reason

            # 3. Finite value check (NaN / Inf)
            if torch.isnan(v).any() or torch.isinf(v).any():
                reason = f"Non-finite values (NaN/Inf) detected in '{k}'"
                self._log_rejection(client_id, current_round, reason)
                return False, None, reason

        # 4. Duplicate update detection via SHA-256 fingerprint
        payload_bytes = serialize_model_state(client_state)
        update_hash = hashlib.sha256(payload_bytes).hexdigest()
        
        if self.seen_update_hashes.get(client_id) == update_hash:
            reason = "Duplicate/replay update detected from same client"
            self._log_rejection(client_id, current_round, reason)
            return False, None, reason
            
        self.seen_update_hashes[client_id] = update_hash

        # 5. L2 Update Norm Clipping: Delta = W_client - W_global
        # 5. L2 Update Norm Clipping on floating weights only (excluding BatchNorm running buffers)
        total_norm_sq = 0.0
        deltas = {}
        for k, v in client_state.items():
            if v.is_floating_point() and "running_" not in k and "num_batches" not in k:
                d = v.to(torch.float32) - global_state[k].to(torch.float32)
                deltas[k] = d
                total_norm_sq += torch.sum(d ** 2).item()
            else:
                deltas[k] = None

        l2_norm = float(total_norm_sq ** 0.5)
        clip_coef = min(1.0, self.max_norm_clip / (l2_norm + 1e-6))

        # Reconstruct bounded update: W_bounded = W_global + clip_coef * Delta
        bounded_state: Dict[str, torch.Tensor] = {}
        for k, v in global_state.items():
            if deltas[k] is not None:
                bounded_state[k] = v.to(torch.float32) + (clip_coef * deltas[k])
            else:
                # Retain client running statistics with preserved dtype
                bounded_state[k] = client_state[k].clone()

        return True, bounded_state, f"Accepted (L2 norm={l2_norm:.3f}, clip_coef={clip_coef:.3f})"

    def _log_rejection(self, client_id: int, current_round: int, reason: str):
        self.rejection_log.append({
            "client_id": client_id,
            "round": current_round,
            "reason": reason
        })

    def get_rejection_summary(self) -> List[Dict[str, Any]]:
        return list(self.rejection_log)
