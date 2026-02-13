from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from core.strategy import HardRuleConfig, evaluate_hard_rule_short


class StrategyShortRuleTest(unittest.TestCase):
    def test_short_rule_matches_overbought_signal(self) -> None:
        now = datetime.now(timezone.utc)
        rows = []
        price = 100.0
        for i in range(80):
            price += 0.6
            rows.append(
                {
                    "time": now + timedelta(minutes=i),
                    "open": price - 0.2,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000 + i,
                }
            )
        frame = pd.DataFrame(rows).set_index("time")
        cfg = HardRuleConfig(required_signal_count=1, use_macd_dead_cross=False)
        result = evaluate_hard_rule_short(frame, config=cfg)

        self.assertTrue(result.is_buy_candidate)
        self.assertGreaterEqual(result.score, 1)
        self.assertIn("rsi_overbought", result.reasons)


if __name__ == "__main__":
    unittest.main()
