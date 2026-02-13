from __future__ import annotations

import unittest

from data.collector import MarketDataCollector


class _FakeExchange:
    quote_currency = "USDT"

    def get_top_volume_tickers(self, limit: int = 10, quote: str = "USDT"):
        return ["BTCUSDT", "ETHUSDT", "XRPUSDT"][:limit]

    def normalize_market(self, ticker: str) -> str:
        value = ticker.upper().strip()
        if value.endswith(self.quote_currency):
            return value
        return f"{value}{self.quote_currency}"

    def get_ohlcv(self, ticker: str, interval: str = "minute5", count: int = 200):
        return None

    def get_orderbook(self, ticker: str):
        return None


class CollectorWatchlistTest(unittest.TestCase):
    def test_refresh_watchlist_merges_top_and_extra(self) -> None:
        collector = MarketDataCollector(
            exchange=_FakeExchange(),
            top_n=3,
            max_watchlist=5,
            extra_watchlist=("SOLUSDT", "ada", "XRPUSDT"),
        )
        watchlist = collector.refresh_watchlist()
        self.assertEqual(watchlist, ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"])


if __name__ == "__main__":
    unittest.main()
