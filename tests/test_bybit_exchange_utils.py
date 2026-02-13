from __future__ import annotations

import unittest

from core.bybit_exchange import BybitExchange


class BybitExchangeUtilsTest(unittest.TestCase):
    def test_normalize_order_qty_with_step(self) -> None:
        ex = BybitExchange(enable_trading=False)
        ex._instrument_cache["BTCUSDT"] = {  # type: ignore[attr-defined]
            "qty_step": 0.001,
            "min_qty": 0.01,
            "max_qty": 0.0,
            "min_notional": 0.0,
        }
        out = ex._normalize_order_qty("BTCUSDT", qty=0.01234, notional=None)  # type: ignore[attr-defined]
        self.assertAlmostEqual(out, 0.012, places=9)

    def test_normalize_order_qty_from_notional(self) -> None:
        ex = BybitExchange(enable_trading=False)
        ex._instrument_cache["BTCUSDT"] = {  # type: ignore[attr-defined]
            "qty_step": 0.001,
            "min_qty": 0.005,
            "max_qty": 0.0,
            "min_notional": 0.0,
        }
        ex.get_current_price = lambda _ticker: 2000.0  # type: ignore[assignment]
        out = ex._normalize_order_qty("BTCUSDT", qty=0.0, notional=25.0)  # type: ignore[attr-defined]
        self.assertAlmostEqual(out, 0.012, places=9)


if __name__ == "__main__":
    unittest.main()
