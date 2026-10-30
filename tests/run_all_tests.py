"""
Master Test Runner for AIORI Hackathon 3 PS-003.
Runs the entire test suite across all development phases up to Phase 3:
- Phase 0: Workspace, Hardware, Environment
- Phase 1: Dataset, Provenance, Adjacency Graph, Chronological Splitting, IID/Non-IID Partitioning
- Phase 2: CompactSTGCN Architecture, Graph Convolutions, Node-Invariance, Gradient Stability
- Phase 3: Centralized Baseline Training, Convergence Logs, Checkpoint Integrity, Persistence Benchmark
"""

import os
import sys
import time
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_dataset import TestPEMS04Dataset
from tests.test_model import TestCompactSTGCN
from tests.test_centralized import TestCentralizedBaseline
from tests.test_fedavg import TestFedAvg
from tests.test_budget import TestBudgetAndScheduler
from tests.test_validation import TestValidationAndBEFA


def main():
    print("\n" + "=" * 80)
    print(" AIORI PS-003: MASTER TEST SUITE (PHASES 0, 1, 2, 3, 4, AND 5)")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases across all phases
    suite.addTests(loader.loadTestsFromTestCase(TestPEMS04Dataset))
    suite.addTests(loader.loadTestsFromTestCase(TestCompactSTGCN))
    suite.addTests(loader.loadTestsFromTestCase(TestCentralizedBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestFedAvg))
    suite.addTests(loader.loadTestsFromTestCase(TestBudgetAndScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationAndBEFA))

    print(f"\n[Test Discovery] Loaded {suite.countTestCases()} unit tests across 6 suites:")
    print("  - Suite 1 (Phase 0 & 1): TestPEMS04Dataset         (7 Tests)")
    print("  - Suite 2 (Phase 2):     TestCompactSTGCN          (7 Tests)")
    print("  - Suite 3 (Phase 3):     TestCentralizedBaseline   (5 Tests)")
    print("  - Suite 4 (Phase 4):     TestFedAvg                (7 Tests)")
    print("  - Suite 5 (Phase 5):     TestBudgetAndScheduler    (3 Tests)")
    print("  - Suite 6 (Phase 5):     TestValidationAndBEFA     (5 Tests)")
    print("-" * 80)

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f" EXECUTION SUMMARY: {result.testsRun} Tests Completed in {elapsed:.2f} s")
    print(f" Passed: {result.testsRun - len(result.failures) - len(result.errors)} | Failures: {len(result.failures)} | Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(" VERDICT: ALL PHASES 0, 1, 2, 3, 4, AND 5 TESTS PASSED (100% SUCCESS)")
    else:
        print(" VERDICT: TEST FAILURES DETECTED")
    print("=" * 80)

    return result.wasSuccessful()


run_master_test_suite = main


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
