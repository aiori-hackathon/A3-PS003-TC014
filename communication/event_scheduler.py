"""
Event-Triggered Client Participation Scheduler.
Monitors local traffic changes and validation dynamics to schedule
client communication only when updates provide meaningful learning value.
Enforces maximum update intervals to prevent client starvation.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


class EventScheduler:
    """
    Schedules client participation based on local traffic events and staleness constraints.
    """
    def __init__(
        self,
        num_clients: int = 5,
        event_threshold: float = 0.25,
        max_update_interval: int = 2,
        traffic_weight: float = 0.6,
        loss_weight: float = 0.4
    ):
        self.num_clients = num_clients
        self.event_threshold = event_threshold
        self.max_update_interval = max_update_interval
        self.traffic_weight = traffic_weight
        self.loss_weight = loss_weight

        # State tracking per client
        self.rounds_since_last_update: Dict[int, int] = {i: 0 for i in range(num_clients)}
        self.last_participated_round: Dict[int, int] = {i: 0 for i in range(num_clients)}
        self.baseline_traffic_means: Dict[int, float] = {}
        self.previous_val_losses: Dict[int, float] = {}
        self.event_scores_history: Dict[int, List[float]] = {i: [] for i in range(num_clients)}

    def register_client_baseline(self, client_id: int, initial_mean_flow: float, initial_val_loss: float = 1.0):
        """Initializes client traffic baseline statistics."""
        self.baseline_traffic_means[client_id] = initial_mean_flow
        self.previous_val_losses[client_id] = initial_val_loss

    def compute_event_score(
        self,
        client_id: int,
        current_mean_flow: float,
        current_val_loss: float
    ) -> float:
        """
        Computes normalized local traffic event score E_i in [0, 1+].
        Combines relative traffic deviation and validation loss drift:
          E_i = alpha * (|X - X_base| / X_base) + (1 - alpha) * max(0, (L - L_prev) / L_prev)
        """
        base_flow = self.baseline_traffic_means.get(client_id, current_mean_flow)
        prev_loss = self.previous_val_losses.get(client_id, current_val_loss)

        flow_diff_rel = abs(current_mean_flow - base_flow) / (base_flow + 1e-4)
        loss_drift_rel = max(0.0, (current_val_loss - prev_loss) / (prev_loss + 1e-4))

        score = (self.traffic_weight * flow_diff_rel) + (self.loss_weight * loss_drift_rel)
        self.event_scores_history[client_id].append(float(score))
        return float(score)

    def evaluate_participation(
        self,
        client_id: int,
        current_round: int,
        event_score: float
    ) -> Tuple[bool, str]:
        """
        Determines whether client should participate in the current round.
        Returns (should_participate, reason).
        """
        # Round 1 is the mandatory initial bootstrap/calibration round
        if current_round == 1:
            return True, "bootstrap_initialization (round=1)"

        staleness = self.rounds_since_last_update.get(client_id, 0)

        # Starvation Prevention Fallback: if client hasn't updated in max_update_interval rounds
        if staleness >= self.max_update_interval:
            return True, f"starvation_prevention (staleness={staleness} >= {self.max_update_interval})"

        # Event-Triggered Condition: local traffic state diverged meaningfully
        if event_score >= self.event_threshold:
            return True, f"event_triggered (score={event_score:.3f} >= threshold={self.event_threshold})"

        # Otherwise: conditions are stable, defer communication to conserve budget
        return False, f"deferred_stable (score={event_score:.3f} < threshold={self.event_threshold})"

    def record_round_completion(self, participating_client_ids: List[int], current_round: int):
        """Updates staleness counters at the end of each communication round."""
        for c in range(self.num_clients):
            if c in participating_client_ids:
                self.rounds_since_last_update[c] = 0
                self.last_participated_round[c] = current_round
            else:
                self.rounds_since_last_update[c] += 1
