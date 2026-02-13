"""Bithumb API 래퍼."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency import guard
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency import guard
    yaml = None

try:
    import pybithumb
except Exception:  # pragma: no cover - optional dependency import guard
    pybithumb = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExchangeCredentials:
    """API credential container."""

    api_key: str = ""
    secret_key: str = ""

    @property
    def is_ready(self) -> bool:
        return bool(self.api_key and self.secret_key)


def load_credentials(config_path: str = "config.yaml") -> ExchangeCredentials:
    """Load credentials from environment first, then optional yaml config."""
    load_dotenv()

    api_key = os.getenv("BITHUMB_API_KEY", "").strip()
    secret_key = os.getenv("BITHUMB_SECRET_KEY", "").strip()

    if api_key and secret_key:
        return ExchangeCredentials(api_key=api_key, secret_key=secret_key)

    cfg_file = Path(config_path)
    if cfg_file.exists() and yaml is not None:
        try:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            api_key = (cfg.get("BITHUMB_API_KEY") or cfg.get("api_key") or "").strip()
            secret_key = (cfg.get("BITHUMB_SECRET_KEY") or cfg.get("secret_key") or "").strip()
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", config_path, exc)

    return ExchangeCredentials(api_key=api_key, secret_key=secret_key)


class BithumbExchange:
    """Thin wrapper around pybithumb for public/private endpoints."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        public_retry_count: int = 2,
        public_retry_delay_sec: float = 0.2,
    ) -> None:
        self.credentials = load_credentials(config_path=config_path)
        self.public_retry_count = public_retry_count
        self.public_retry_delay_sec = public_retry_delay_sec
        self._private_client = None
        if pybithumb is None:
            LOGGER.warning("pybithumb is not installed. Exchange calls will be disabled.")
            return
        if self.credentials.is_ready:
            self._private_client = pybithumb.Bithumb(
                self.credentials.api_key,
                self.credentials.secret_key,
            )

    @property
    def trading_enabled(self) -> bool:
        return self._private_client is not None

    def connectivity_report(self, sample_ticker: str = "BTC") -> dict[str, Any]:
        """Return public/private API connectivity summary.

        The report is designed for Step 1 account linkage checks.
        """
        report: dict[str, Any] = {
            "pybithumb_installed": pybithumb is not None,
            "credentials_loaded": self.credentials.is_ready,
            "trading_enabled": self.trading_enabled,
            "public_api_ok": False,
            "private_api_ok": False,
            "sample_ticker": sample_ticker,
            "sample_price": None,
            "sample_balance": None,
            "error": None,
        }

        price = self.get_current_price(sample_ticker)
        if price is not None:
            report["public_api_ok"] = True
            report["sample_price"] = price

        if not self.trading_enabled:
            return report

        try:
            report["sample_balance"] = self._private_client.get_balance(sample_ticker)
            report["private_api_ok"] = True
        except Exception as exc:
            report["error"] = str(exc)

        return report

    def get_ohlcv(self, ticker: str, interval: str = "minute5", count: int = 200):
        """Fetch OHLCV dataframe from Bithumb."""
        if pybithumb is None:
            return None

        def _fetch():
            frame = pybithumb.get_ohlcv(ticker, interval=interval)
            if frame is None:
                return None
            return frame.tail(count).copy()

        return self._retry_public(_fetch, f"OHLCV {ticker}:{interval}")

    def get_orderbook(self, ticker: str) -> dict[str, Any] | None:
        """Fetch orderbook snapshot."""
        if pybithumb is None:
            return None
        return self._retry_public(lambda: pybithumb.get_orderbook(ticker), f"orderbook {ticker}")

    def get_current_price(self, ticker: str) -> float | None:
        """Fetch current price."""
        if pybithumb is None:
            return None

        def _fetch():
            price = pybithumb.get_current_price(ticker)
            return float(price) if price is not None else None

        return self._retry_public(_fetch, f"current_price {ticker}")

    def get_balance(self, ticker: str) -> Any:
        """Fetch balance for a given ticker/currency."""
        if not self.trading_enabled:
            raise RuntimeError("Private API disabled. Check Bithumb API key/secret.")
        return self._private_client.get_balance(ticker)

    def market_buy(self, ticker: str, quantity: float) -> Any:
        """Place market buy order."""
        if not self.trading_enabled:
            raise RuntimeError("Private API disabled. Check Bithumb API key/secret.")
        return self._private_client.buy_market_order(ticker, quantity)

    def market_sell(self, ticker: str, quantity: float) -> Any:
        """Place market sell order."""
        if not self.trading_enabled:
            raise RuntimeError("Private API disabled. Check Bithumb API key/secret.")
        return self._private_client.sell_market_order(ticker, quantity)

    def get_top_volume_tickers(self, limit: int = 5, quote: str = "KRW") -> list[str]:
        """Return top tickers by recent quote volume."""
        if pybithumb is None:
            return []
        scored: list[tuple[float, str]] = []
        try:
            tickers = pybithumb.get_tickers(payment_currency=quote) or []
            for ticker in tickers:
                frame = self.get_ohlcv(ticker=ticker, interval="minute60", count=24)
                if frame is None or frame.empty:
                    continue
                volume = (frame["close"] * frame["volume"]).sum()
                scored.append((float(volume), ticker))
        except Exception as exc:
            LOGGER.warning("Top volume ticker scan failed: %s", exc)
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ticker for _, ticker in scored[:limit]]

    def _retry_public(self, fn, label: str):
        attempts = max(1, self.public_retry_count)
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                LOGGER.warning("Public API failed (%s) attempt=%s/%s: %s", label, attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(self.public_retry_delay_sec)
        if last_exc is not None:
            LOGGER.warning("Public API exhausted retries (%s): %s", label, last_exc)
        return None
