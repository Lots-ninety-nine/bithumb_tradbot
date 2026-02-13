from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from core.advanced_signals import evaluate_advanced_signals
from core.config_loader import AdvancedSignalConfig


class AdvancedSignalsTest(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        rows = []
        base = 100.0
        now = datetime.now(timezone.utc)
        for i in range(150):
            close = base + (i * 0.05)
            rows.append(
                {
                    "time": now,
                    "open": close - 0.2,
                    "high": close + 0.6,
                    "low": close - 0.8,
                    "close": close,
                    "volume": 1000 + i,
                }
            )
        frame = pd.DataFrame(rows).set_index("time")
        return frame

    def test_evaluate_advanced_signals_returns_metrics(self) -> None:
        frame = self._sample_frame()
        orderbook = {
            "orderbook_units": [
                {
                    "ask_price": 101.0,
                    "bid_price": 100.8,
                    "ask_size": 10.0,
                    "bid_size": 14.0,
                }
                for _ in range(5)
            ]
        }
        result = evaluate_advanced_signals(
            frame=frame,
            orderbook=orderbook,
            config=AdvancedSignalConfig(),
        )
        self.assertIsInstance(result.total_score, float)
        self.assertIsInstance(result.reasons, list)
        self.assertIn("weighted_score", result.metrics)


if __name__ == "__main__":
    unittest.main()
