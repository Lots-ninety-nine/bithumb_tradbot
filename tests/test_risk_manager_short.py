from __future__ import annotations

import unittest

from core.risk_manager import RiskManager


class RiskManagerShortTest(unittest.TestCase):
    def test_short_trailing_stop_exit(self) -> None:
        risk = RiskManager(
            seed_krw=100000,
            slot_count=1,
            stop_loss_pct=0.03,
            trailing_start_pct=0.01,
            trailing_gap_pct=0.015,
        )
        position = risk.open_position(
            ticker="BTCUSDT",
            quantity=0.01,
            entry_price=100.0,
            side="SHORT",
        )
        self.assertIsNotNone(position)
        assert position is not None

        hold = risk.evaluate_exit(position=position, current_price=98.0)
        self.assertFalse(hold.should_exit)
        self.assertEqual(hold.reason, "trailing_active")

        exit_decision = risk.evaluate_exit(position=position, current_price=99.8)
        self.assertTrue(exit_decision.should_exit)
        self.assertEqual(exit_decision.reason, "trailing_stop")


if __name__ == "__main__":
    unittest.main()
