"""시장 데이터 수집기."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class TickerSnapshot:
    """A single ticker snapshot."""

    ticker: str
    updated_at: datetime
    candles: dict[str, Any] = field(default_factory=dict)
    orderbook: dict[str, Any] | None = None


@dataclass(slots=True)
class IntervalQuality:
    """Single interval quality metrics."""

    interval: str
    has_data: bool
    row_count: int
    last_timestamp: str | None
    freshness_sec: float | None
    is_fresh: bool
    is_sufficient_rows: bool


class MarketDataCollector:
    """Collect top-volume ticker OHLCV + orderbook in memory."""

    def __init__(
        self,
        exchange: Any,
        intervals: tuple[str, ...] = ("minute1", "minute5", "minute15"),
        top_n: int = 15,
        max_watchlist: int = 25,
        candle_count: int = 200,
        extra_watchlist: tuple[str, ...] = (),
    ) -> None:
        self.exchange = exchange
        self.intervals = intervals
        self.top_n = top_n
        self.max_watchlist = max_watchlist
        self.candle_count = candle_count
        self.extra_watchlist = list(extra_watchlist)
        self.watchlist: list[str] = []
        self.snapshots: dict[str, TickerSnapshot] = {}

    def refresh_watchlist(self) -> list[str]:
        top_tickers = self.exchange.get_top_volume_tickers(limit=self.top_n)
        merged = self._merge_watchlist(top_tickers, self.extra_watchlist)
        if self.max_watchlist > 0:
            merged = merged[: self.max_watchlist]
        self.watchlist = merged
        return merged

    def collect_once(self) -> dict[str, TickerSnapshot]:
        """Collect one full pass for the current watchlist."""
        if not self.watchlist:
            self.refresh_watchlist()

        now = datetime.now(timezone.utc)
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

    def _merge_watchlist(self, top_tickers: list[str], extra_watchlist: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for ticker in [*top_tickers, *extra_watchlist]:
            norm = self.exchange.normalize_market(ticker).upper()
            if norm in seen:
                continue
            seen.add(norm)
            merged.append(norm)
        return merged

    def get_data_quality_report(self, min_rows: int = 60) -> dict[str, list[dict[str, Any]]]:
        """Return lightweight data quality report for collected snapshots."""
        if not self.snapshots:
            self.collect_once()

        report: dict[str, list[dict[str, Any]]] = {}
        for ticker, snapshot in self.snapshots.items():
            per_interval: list[dict[str, Any]] = []
            for interval in self.intervals:
                frame = snapshot.candles.get(interval)
                quality = self._measure_interval_quality(
                    frame=frame,
                    interval=interval,
                    now=snapshot.updated_at,
                    min_rows=min_rows,
                )
                per_interval.append(asdict(quality))
            report[ticker] = per_interval
        return report

    def _measure_interval_quality(
        self,
        frame: Any,
        interval: str,
        now: datetime,
        min_rows: int,
    ) -> IntervalQuality:
        if frame is None or getattr(frame, "empty", True):
            return IntervalQuality(
                interval=interval,
                has_data=False,
                row_count=0,
                last_timestamp=None,
                freshness_sec=None,
                is_fresh=False,
                is_sufficient_rows=False,
            )

        row_count = int(len(frame))
        last_ts = self._to_datetime(frame.index[-1])
        freshness = None
        is_fresh = False
        if last_ts is not None:
            freshness = max(0.0, (self._as_utc(now) - self._as_utc(last_ts)).total_seconds())
            is_fresh = freshness <= self._freshness_threshold_sec(interval)

        return IntervalQuality(
            interval=interval,
            has_data=True,
            row_count=row_count,
            last_timestamp=last_ts.isoformat() if last_ts else None,
            freshness_sec=freshness,
            is_fresh=is_fresh,
            is_sufficient_rows=row_count >= min_rows,
        )

    @staticmethod
    def _to_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime()
            except Exception:
                return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _as_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _freshness_threshold_sec(interval: str) -> int:
        mapping = {
            "minute1": 3 * 60,
            "minute3": 7 * 60,
            "minute5": 12 * 60,
            "minute10": 20 * 60,
            "minute15": 30 * 60,
            "minute30": 50 * 60,
            "minute60": 90 * 60,
        }
        return mapping.get(interval, 15 * 60)
