"""Bybit V5 exchange wrapper for linear perpetual long/short trading."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class BybitCredentials:
    api_key: str = ""
    secret_key: str = ""

    @property
    def is_ready(self) -> bool:
        return bool(self.api_key and self.secret_key)


def load_bybit_credentials() -> BybitCredentials:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
    return BybitCredentials(
        api_key=os.getenv("BYBIT_API_KEY", "").strip(),
        secret_key=os.getenv("BYBIT_API_SECRET", "").strip(),
    )


class BybitExchange:
    """Bybit V5 linear perpetual adapter with Bithumb-like interface."""

    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        category: str = "linear",
        quote_coin: str = "USDT",
        account_type: str = "UNIFIED",
        recv_window: int = 5000,
        leverage: float = 2.0,
        public_retry_count: int = 2,
        public_retry_delay_sec: float = 0.2,
        timeout_sec: float = 8.0,
        enable_trading: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.category = category
        self.quote_coin = quote_coin.upper()
        self.account_type = account_type
        self.recv_window = int(recv_window)
        self.leverage = float(leverage)
        self.public_retry_count = int(public_retry_count)
        self.public_retry_delay_sec = float(public_retry_delay_sec)
        self.timeout_sec = float(timeout_sec)
        self.enable_trading = bool(enable_trading)
        self.credentials = load_bybit_credentials()
        self._session = requests.Session() if requests is not None else None
        self._api_usage: dict[str, Any] = {
            "public": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
            "private": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
        }
        self._leverage_cache: set[str] = set()

    @property
    def public_api_enabled(self) -> bool:
        return self._session is not None

    @property
    def private_api_enabled(self) -> bool:
        return self.public_api_enabled and self.credentials.is_ready

    @property
    def trading_enabled(self) -> bool:
        return self.private_api_enabled and self.enable_trading

    @property
    def supports_short(self) -> bool:
        return True

    @property
    def quote_currency(self) -> str:
        return self.quote_coin

    def connectivity_report(self, sample_ticker: str = "BTCUSDT") -> dict[str, Any]:
        report: dict[str, Any] = {
            "provider": "bybit",
            "base_url": self.base_url,
            "requests_installed": requests is not None,
            "pandas_installed": pd is not None,
            "credentials_loaded": self.credentials.is_ready,
            "public_api_enabled": self.public_api_enabled,
            "private_api_enabled": self.private_api_enabled,
            "trading_enabled": self.trading_enabled,
            "sample_ticker": sample_ticker,
            "sample_market": self.normalize_market(sample_ticker),
            "sample_price": None,
            "sample_balance": None,
            "error": None,
        }
        try:
            report["sample_price"] = self.get_current_price(sample_ticker)
            report["sample_balance"] = self.get_available_krw()
        except Exception as exc:
            report["error"] = str(exc)
        return report

    def get_market_codes(self, quote: str | None = None) -> list[str]:
        quote_coin = (quote or self.quote_coin).upper()
        cursor = ""
        out: list[str] = []
        while True:
            params = {
                "category": self.category,
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._public_get("/v5/market/instruments-info", params=params)
            rows = self._extract_result_list(data)
            for row in rows:
                symbol = str(row.get("symbol", "")).upper()
                if not symbol:
                    continue
                if str(row.get("quoteCoin", "")).upper() != quote_coin:
                    continue
                if str(row.get("status", "")).lower() not in {"trading", "settling"}:
                    continue
                out.append(symbol)
            cursor = str(((data or {}).get("result") or {}).get("nextPageCursor") or "")
            if not cursor:
                break
        return sorted(set(out))

    def get_top_volume_tickers(self, limit: int = 20, quote: str = "USDT") -> list[str]:
        data = self._public_get("/v5/market/tickers", params={"category": self.category})
        rows = self._extract_result_list(data)
        quote_coin = quote.upper()
        scored: list[tuple[float, str]] = []
        for row in rows:
            symbol = str(row.get("symbol", "")).upper()
            if not symbol.endswith(quote_coin):
                continue
            turnover = self._to_float(row.get("turnover24h"))
            if turnover <= 0:
                continue
            scored.append((turnover, symbol))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def normalize_market(self, ticker: str) -> str:
        value = ticker.upper().strip()
        if "-" in value:
            value = value.split("-", 1)[1]
        if value.endswith(self.quote_coin):
            return value
        return f"{value}{self.quote_coin}"

    def get_ohlcv(self, ticker: str, interval: str = "minute5", count: int = 200):
        if pd is None:
            return None
        symbol = self.normalize_market(ticker)
        bybit_interval = self._map_interval(interval)
        data = self._public_get(
            "/v5/market/kline",
            params={
                "category": self.category,
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": max(20, min(1000, int(count))),
            },
        )
        rows = self._extract_result_list(data)
        if not rows:
            return pd.DataFrame()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 7:
                continue
            ts_ms = int(float(row[0]))
            parsed.append(
                {
                    "time": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "open": self._to_float(row[1]),
                    "high": self._to_float(row[2]),
                    "low": self._to_float(row[3]),
                    "close": self._to_float(row[4]),
                    "volume": self._to_float(row[5]),
                    "value": self._to_float(row[6]),
                }
            )
        frame = pd.DataFrame(parsed)
        if frame.empty:
            return frame
        frame = frame.set_index("time").sort_index()
        return frame.tail(count).copy()

    def get_orderbook(self, ticker: str) -> dict[str, Any] | None:
        symbol = self.normalize_market(ticker)
        data = self._public_get(
            "/v5/market/orderbook",
            params={
                "category": self.category,
                "symbol": symbol,
                "limit": 50,
            },
        )
        result = ((data or {}).get("result") or {}) if isinstance(data, dict) else {}
        bids = result.get("b") if isinstance(result, dict) else None
        asks = result.get("a") if isinstance(result, dict) else None
        if not isinstance(bids, list) or not isinstance(asks, list):
            return None

        units: list[dict[str, float]] = []
        depth = min(10, len(bids), len(asks))
        for i in range(depth):
            bid = bids[i]
            ask = asks[i]
            if not isinstance(bid, list) or not isinstance(ask, list):
                continue
            units.append(
                {
                    "bid_price": self._to_float(bid[0]),
                    "bid_size": self._to_float(bid[1]),
                    "ask_price": self._to_float(ask[0]),
                    "ask_size": self._to_float(ask[1]),
                }
            )
        return {
            "symbol": symbol,
            "orderbook_units": units,
        }

    def get_current_price(self, ticker: str) -> float | None:
        symbol = self.normalize_market(ticker)
        data = self._public_get(
            "/v5/market/tickers",
            params={"category": self.category, "symbol": symbol},
        )
        rows = self._extract_result_list(data)
        if not rows:
            return None
        last = rows[0]
        return self._to_float(last.get("lastPrice"))

    def get_accounts(self) -> list[dict[str, Any]]:
        result = self._private_get(
            "/v5/account/wallet-balance",
            params={"accountType": self.account_type, "coin": self.quote_coin},
        )
        rows = self._extract_result_list(result)
        if not rows:
            return []
        wallet = rows[0]
        coin_rows = wallet.get("coin", []) if isinstance(wallet, dict) else []
        if isinstance(coin_rows, list):
            return [row for row in coin_rows if isinstance(row, dict)]
        return []

    def get_available_krw(self) -> float | None:
        accounts = self.get_accounts()
        for row in accounts:
            coin = str(row.get("coin", "")).upper()
            if coin != self.quote_coin:
                continue
            available = self._to_float(row.get("availableToWithdraw"))
            if available > 0:
                return available
            available = self._to_float(row.get("walletBalance"))
            if available > 0:
                return available
        return None

    def get_total_asset_krw(self) -> float | None:
        """Compatibility: returns total equity in quote currency (USDT)."""
        result = self._private_get(
            "/v5/account/wallet-balance",
            params={"accountType": self.account_type},
        )
        rows = self._extract_result_list(result)
        if not rows:
            return None
        total_equity = self._to_float(rows[0].get("totalEquity"))
        return total_equity if total_equity > 0 else None

    def get_api_usage_snapshot(self, reset: bool = False) -> dict[str, Any]:
        snapshot = deepcopy(self._api_usage)
        if reset:
            self._api_usage = {
                "public": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
                "private": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
            }
        return snapshot

    def get_bithumb_notices(self, page: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        return []

    def execute_entry(
        self,
        ticker: str,
        side: str,
        quantity: float,
        notional: float | None = None,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 0.0,
        order_fill_poll_sec: float = 0.0,
        cancel_unfilled_before_retry: bool = False,
    ) -> dict[str, Any] | None:
        _ = (notional, order_fill_wait_sec, order_fill_poll_sec, cancel_unfilled_before_retry)
        symbol = self.normalize_market(ticker)
        bybit_side = "Buy" if side.upper() == "LONG" else "Sell"
        self._ensure_leverage(symbol)
        return self._retry_order(
            fn=lambda: self._place_order(symbol=symbol, side=bybit_side, qty=quantity, reduce_only=False),
            retry_count=order_retry_count,
            retry_delay_sec=order_retry_delay_sec,
        )

    def execute_exit(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 0.0,
        order_fill_poll_sec: float = 0.0,
        cancel_unfilled_before_retry: bool = False,
    ) -> dict[str, Any] | None:
        _ = (order_fill_wait_sec, order_fill_poll_sec, cancel_unfilled_before_retry)
        symbol = self.normalize_market(ticker)
        bybit_side = "Sell" if side.upper() == "LONG" else "Buy"
        return self._retry_order(
            fn=lambda: self._place_order(symbol=symbol, side=bybit_side, qty=quantity, reduce_only=True),
            retry_count=order_retry_count,
            retry_delay_sec=order_retry_delay_sec,
        )

    def _retry_order(self, fn, retry_count: int, retry_delay_sec: float):
        attempts = max(1, int(retry_count))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(retry_delay_sec)))
        raise RuntimeError(f"Bybit order failed after retries: {last_exc}")

    def _ensure_leverage(self, symbol: str) -> None:
        if symbol in self._leverage_cache:
            return
        if self.leverage <= 0:
            return
        payload = {
            "category": self.category,
            "symbol": symbol,
            "buyLeverage": self._fmt(self.leverage),
            "sellLeverage": self._fmt(self.leverage),
        }
        try:
            self._private_post("/v5/position/set-leverage", payload)
            self._leverage_cache.add(symbol)
        except Exception as exc:
            LOGGER.warning("Bybit set leverage failed symbol=%s err=%s", symbol, exc)

    def _place_order(self, symbol: str, side: str, qty: float, reduce_only: bool) -> dict[str, Any] | None:
        payload = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": self._fmt(qty, precision=6),
            "reduceOnly": reduce_only,
            "timeInForce": "IOC",
        }
        return self._private_post("/v5/order/create", payload)

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        def _fetch():
            self._mark_api_usage(kind="public", path=path, success=False, before_call=True)
            payload = self._request("GET", path, params=params, signed=False)
            self._mark_api_usage(kind="public", path=path, success=True, before_call=False)
            return payload

        return self._retry_public(_fetch, f"GET {path}")

    def _private_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self._mark_api_usage(kind="private", path=path, success=False, before_call=True)
        try:
            payload = self._request("GET", path, params=params, signed=True)
            self._mark_api_usage(kind="private", path=path, success=True, before_call=False)
            return payload
        except Exception:
            self._mark_api_usage(kind="private", path=path, success=False, before_call=False)
            raise

    def _private_post(self, path: str, body: dict[str, Any]) -> Any:
        self._mark_api_usage(kind="private", path=path, success=False, before_call=True)
        try:
            payload = self._request("POST", path, body=body, signed=True)
            self._mark_api_usage(kind="private", path=path, success=True, before_call=False)
            return payload
        except Exception:
            self._mark_api_usage(kind="private", path=path, success=False, before_call=False)
            raise

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("requests is not installed")
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        req_params = params or {}
        req_body = body or {}
        if signed:
            if not self.credentials.is_ready:
                raise RuntimeError("Bybit credentials are missing")
            ts = str(int(time.time() * 1000))
            recv = str(self.recv_window)
            raw = urlencode(req_params, doseq=True) if method.upper() == "GET" else json.dumps(req_body, separators=(",", ":"))
            sign_payload = f"{ts}{self.credentials.api_key}{recv}{raw}"
            sign = hmac.new(
                self.credentials.secret_key.encode("utf-8"),
                sign_payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers.update(
                {
                    "X-BAPI-API-KEY": self.credentials.api_key,
                    "X-BAPI-TIMESTAMP": ts,
                    "X-BAPI-RECV-WINDOW": recv,
                    "X-BAPI-SIGN": sign,
                    "X-BAPI-SIGN-TYPE": "2",
                }
            )

        response = self._session.request(
            method=method.upper(),
            url=url,
            params=req_params if req_params else None,
            json=req_body if method.upper() != "GET" else None,
            headers=headers,
            timeout=self.timeout_sec,
        )
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON response status={response.status_code}: {exc}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected response: {payload}")
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"API error: {payload.get('retCode')} {payload.get('retMsg')}")
        return payload

    def _retry_public(self, fn, label: str):
        attempts = max(1, int(self.public_retry_count))
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                kind, path = self._extract_kind_path_from_label(label)
                self._mark_api_usage(kind=kind, path=path, success=False, before_call=False)
                LOGGER.warning("Public API failed (%s) attempt=%s/%s: %s", label, attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(self.public_retry_delay_sec)
        if last_exc is not None:
            LOGGER.warning("Public API exhausted retries (%s): %s", label, last_exc)
        return None

    def _mark_api_usage(self, kind: str, path: str, success: bool, before_call: bool) -> None:
        bucket = self._api_usage.get(kind)
        if not isinstance(bucket, dict):
            return
        by_path = bucket.setdefault("by_path", {})
        stat = by_path.setdefault(path, {"total": 0, "success": 0, "fail": 0})
        if before_call:
            bucket["total"] += 1
            stat["total"] += 1
            return
        if success:
            bucket["success"] += 1
            stat["success"] += 1
        else:
            bucket["fail"] += 1
            stat["fail"] += 1

    @staticmethod
    def _extract_kind_path_from_label(label: str) -> tuple[str, str]:
        if label.startswith("GET "):
            return "public", label.replace("GET ", "", 1).strip()
        return "public", label

    @staticmethod
    def _extract_result_list(payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        rows = result.get("list")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict) or isinstance(row, list)]
        return []

    @staticmethod
    def _map_interval(interval: str) -> str:
        value = interval.lower().strip()
        mapping = {
            "minute1": "1",
            "minute3": "3",
            "minute5": "5",
            "minute15": "15",
            "minute30": "30",
            "minute60": "60",
            "day": "D",
            "week": "W",
            "month": "M",
        }
        return mapping.get(value, "5")

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _fmt(value: float, precision: int = 8) -> str:
        return format(float(value), f".{precision}f").rstrip("0").rstrip(".")
