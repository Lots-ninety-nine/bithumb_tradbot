"""시장 데이터 수집기."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.exchange import BithumbExchange


@dataclass(slots=True)
class TickerSnapshot:
    """A single ticker snapshot."""

    ticker: str
    updated_at: datetime
    candles: dict[str, Any] = field(default_factory=dict)
    orderbook: dict[str, Any] | None = None


class MarketDataCollector:
    """Collect top-volume ticker OHLCV + orderbook in memory."""

    def __init__(
        self,
        exchange: BithumbExchange,
        intervals: tuple[str, ...] = ("minute1", "minute5"),
        top_n: int = 5,
        candle_count: int = 200,
    ) -> None:
        self.exchange = exchange
        self.intervals = intervals
        self.top_n = top_n
        self.candle_count = candle_count
        self.watchlist: list[str] = []
        self.snapshots: dict[str, TickerSnapshot] = {}

    def refresh_watchlist(self) -> list[str]:
        tickers = self.exchange.get_top_volume_tickers(limit=self.top_n)
        self.watchlist = tickers
        return tickers

    def collect_once(self) -> dict[str, TickerSnapshot]:
        """Collect one full pass for the current watchlist."""
        if not self.watchlist:
            self.refresh_watchlist()

        now = datetime.utcnow()
        result: dict[str, TickerSnapshot] = {}
        for ticker in self.watchlist:
            candles: dict[str, Any] = {}
            for interval in self.intervals:
                candles[interval] = self.exchange.get_ohlcv(
                    ticker=ticker,
                    interval=interval,
                    count=self.candle_count,
                )
            snapshot = TickerSnapshot(
                ticker=ticker,
                updated_at=now,
                candles=candles,
                orderbook=self.exchange.get_orderbook(ticker),
            )
            result[ticker] = snapshot

        self.snapshots = result
        return result
