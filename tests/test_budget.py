"""
Unit tests for Phase 5: Communication Budget Manager and Event-Triggered Scheduler.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from communication.budget_manager import CommunicationBudgetManager
from communication.event_scheduler import EventScheduler


class TestBudgetAndScheduler(unittest.TestCase):

    def test_01_budget_allocation_and_enforcement(self):
        """Verify budget tracks byte expenditures and strictly rejects transmissions exceeding quota."""
        total_budget = 1000
        overhead = 50
        mgr = CommunicationBudgetManager(total_budget_bytes=total_budget, protocol_overhead_bytes=overhead)

        self.assertEqual(mgr.get_remaining_budget(), 1000)
        self.assertFalse(mgr.is_exhausted())

        # Request 1: down=200, up=200 -> total = 200 + 200 + 2*50 = 500 bytes (Fits)
        ok = mgr.request_transmission(download_bytes=200, upload_bytes=200)
        self.assertTrue(ok)
        self.assertEqual(mgr.total_used_bytes, 500)
        self.assertEqual(mgr.get_remaining_budget(), 500)

        # Request 2: down=300, up=300 -> total = 600 + 100 = 700 bytes (Exceeds remaining 500 -> Rejected)
        ok2 = mgr.request_transmission(download_bytes=300, upload_bytes=300)
        self.assertFalse(ok2, "Budget manager must reject transactions that exceed budget")
        self.assertEqual(mgr.deferred_transmissions_count, 1)
        self.assertEqual(mgr.get_remaining_budget(), 500)  # Budget preserved

        # Request 3: down=150, up=150 -> total = 300 + 100 = 400 bytes (Fits within 500)
        ok3 = mgr.request_transmission(download_bytes=150, upload_bytes=150)
        self.assertTrue(ok3)
        self.assertEqual(mgr.total_used_bytes, 900)
        self.assertEqual(mgr.get_remaining_budget(), 100)

        # Request 4: down=50, up=50 -> total = 100 + 100 = 200 bytes (Exceeds remaining 100 -> Rejected)
        ok4 = mgr.request_transmission(download_bytes=50, upload_bytes=50)
        self.assertFalse(ok4)
        self.assertEqual(mgr.deferred_transmissions_count, 2)

    def test_02_event_scheduler_triggering_and_deferral(self):
        """Verify event scheduler triggers participation on significant traffic deviations and defers when stable."""
        scheduler = EventScheduler(num_clients=3, event_threshold=0.25, max_update_interval=3)
        
        # Register baseline for client 0 with mean flow = 100.0
        scheduler.register_client_baseline(client_id=0, initial_mean_flow=100.0, initial_val_loss=0.5)

        # Case A: Minimal traffic change (flow = 102.0, loss = 0.5) -> Event score low -> Deferred on round 2
        score_low = scheduler.compute_event_score(client_id=0, current_mean_flow=102.0, current_val_loss=0.5)
        should_part, reason = scheduler.evaluate_participation(client_id=0, current_round=2, event_score=score_low)
        self.assertFalse(should_part, "Client should be deferred when traffic conditions are stable")
        self.assertIn("deferred", reason)

        # Case B: Significant traffic surge (flow = 160.0, loss = 0.8) -> Event score high -> Triggered on round 2
        score_high = scheduler.compute_event_score(client_id=0, current_mean_flow=160.0, current_val_loss=0.8)
        should_part2, reason2 = scheduler.evaluate_participation(client_id=0, current_round=2, event_score=score_high)
        self.assertTrue(should_part2, "Client should be triggered when traffic surges significantly")
        self.assertIn("event_triggered", reason2)

    def test_03_starvation_prevention_fallback(self):
        """Verify that quiet clients are not starved indefinitely and participate once staleness >= max_update_interval."""
        max_interval = 2
        scheduler = EventScheduler(num_clients=2, event_threshold=0.5, max_update_interval=max_interval)
        scheduler.register_client_baseline(client_id=1, initial_mean_flow=200.0)

        # Round 1: Stable conditions -> Deferred
        scheduler.record_round_completion(participating_client_ids=[0], current_round=1)
        self.assertEqual(scheduler.rounds_since_last_update[1], 1)
        part_r1, _ = scheduler.evaluate_participation(client_id=1, current_round=2, event_score=0.1)
        self.assertFalse(part_r1)

        # Round 2: Client 1 still not participating -> Staleness reaches max_interval=2
        scheduler.record_round_completion(participating_client_ids=[0], current_round=2)
        self.assertEqual(scheduler.rounds_since_last_update[1], 2)

        # Round 3: Even with zero event score (0.0), starvation prevention forces participation!
        part_r3, reason_r3 = scheduler.evaluate_participation(client_id=1, current_round=3, event_score=0.0)
        self.assertTrue(part_r3, "Starvation prevention must force participation once max staleness is reached")
        self.assertIn("starvation_prevention", reason_r3)


if __name__ == '__main__':
    unittest.main()
