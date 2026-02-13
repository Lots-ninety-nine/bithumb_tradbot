from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from core.performance_tracker import PerformanceTracker


class PerformanceTrackerTest(unittest.TestCase):
    def test_snapshot_persists_baseline_and_computes_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.json"
            tracker = PerformanceTracker(str(path))
            t0 = datetime(2026, 2, 13, 0, 0, 0, tzinfo=timezone.utc)

            first = tracker.snapshot(current_krw=70000, now=t0)
            self.assertEqual(first.baseline_krw, 70000)
            self.assertAlmostEqual(first.pnl_pct, 0.0, places=6)
            self.assertTrue(path.exists())

            second = tracker.snapshot(current_krw=91000, now=t0)
            self.assertEqual(second.baseline_krw, 70000)
            self.assertAlmostEqual(second.pnl_pct, 30.0, places=6)


if __name__ == "__main__":
    unittest.main()

