"""Bithumb exchange wrapper (REST v2.1.5 first, pybithumb fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
import uuid

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency import guard
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    import jwt
except Exception:  # pragma: no cover - optional dependency import guard
    jwt = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency import guard
    pd = None

try:
    import requests
except Exception:  # pragma: no cover - optional dependency import guard
    requests = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency import guard
    yaml = None

try:
    import pybithumb
except Exception:  # pragma: no cover - optional dependency import guard
    pybithumb = None


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
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
    """Bithumb API client.

    Priority:
    1) Public/private REST API (v2.1.5 docs)
    2) pybithumb fallback for order execution compatibility
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        public_retry_count: int = 2,
        public_retry_delay_sec: float = 0.2,
        timeout_sec: float = 8.0,
        quote_currency: str = "KRW",
        enable_official_orders: bool = False,
    ) -> None:
        self.base_url = os.getenv("BITHUMB_API_BASE_URL", "https://api.bithumb.com").rstrip("/")
        self.credentials = load_credentials(config_path=config_path)
        self.public_retry_count = public_retry_count
        self.public_retry_delay_sec = public_retry_delay_sec
        self.timeout_sec = timeout_sec
        self.quote_currency = quote_currency.upper()
        self.enable_official_orders = enable_official_orders

        self._private_client = None
        self._session = requests.Session() if requests is not None else None
        self._api_usage: dict[str, Any] = {
            "public": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
            "private": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
        }

        if pybithumb is not None and self.credentials.is_ready:
            try:
                self._private_client = pybithumb.Bithumb(
                    self.credentials.api_key,
                    self.credentials.secret_key,
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialize pybithumb private client: %s", exc)

        if requests is None:
            LOGGER.warning("requests is not installed. REST API calls will be disabled.")

    @property
    def public_api_enabled(self) -> bool:
        return self._session is not None

    @property
    def private_api_enabled(self) -> bool:
        return (
            self.public_api_enabled
            and self.credentials.is_ready
            and jwt is not None
        )

    @property
    def trading_enabled(self) -> bool:
        if self._private_client is not None:
            return True
        return self.private_api_enabled and self.enable_official_orders

    @property
    def supports_short(self) -> bool:
        return False

    def connectivity_report(self, sample_ticker: str = "BTC") -> dict[str, Any]:
        """Return public/private API connectivity summary."""
        report: dict[str, Any] = {
            "base_url": self.base_url,
            "requests_installed": requests is not None,
            "pandas_installed": pd is not None,
            "pyjwt_installed": jwt is not None,
            "pybithumb_installed": pybithumb is not None,
            "credentials_loaded": self.credentials.is_ready,
            "public_api_enabled": self.public_api_enabled,
            "private_api_enabled": self.private_api_enabled,
            "trading_enabled": self.trading_enabled,
            "official_orders_enabled": self.enable_official_orders,
            "sample_ticker": sample_ticker,
            "sample_market": self._normalize_market(sample_ticker),
            "sample_price": None,
            "sample_balance": None,
            "error": None,
        }

        try:
            price = self.get_current_price(sample_ticker)
            report["sample_price"] = price
        except Exception as exc:
            report["error"] = str(exc)
            return report

        if not self.private_api_enabled and self._private_client is None:
            return report

        try:
            report["sample_balance"] = self.get_balance(sample_ticker)
        except Exception as exc:
            report["error"] = str(exc)

        return report

    def get_market_codes(self, quote: str | None = None) -> list[str]:
        """Return market ids like KRW-BTC."""
        quote_ccy = (quote or self.quote_currency).upper()
        data = self._public_get("/v1/market/all", {"isDetails": "false"})
        if not isinstance(data, list):
            return []
        markets: list[str] = []
        for row in data:
            market = str(row.get("market", "")).upper()
            if not market:
                continue
            if market.startswith(f"{quote_ccy}-"):
                markets.append(market)
        return markets

    def get_ohlcv(self, ticker: str, interval: str = "minute5", count: int = 200):
        """Fetch OHLCV dataframe from Bithumb REST API."""
        if pd is None:
            return None

        market = self._normalize_market(ticker)
        endpoint, params = self._build_candle_endpoint(market=market, interval=interval, count=count)
        data = self._public_get(endpoint, params)
        frame = self._candles_to_frame(data)
        if frame is None:
            return None
        return frame.tail(count).copy()

    def get_orderbook(self, ticker: str) -> dict[str, Any] | None:
        """Fetch orderbook snapshot."""
        market = self._normalize_market(ticker)
        data = self._public_get("/v1/orderbook", {"markets": market})
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            if isinstance(data.get("orderbook_units"), list):
                return data
            if isinstance(data.get("data"), list):
                return data["data"][0] if data["data"] else None
        return None

    def get_current_price(self, ticker: str) -> float | None:
        """Fetch current trade price."""
        market = self._normalize_market(ticker)
        data = self._public_get("/v1/ticker", {"markets": market})

        row = None
        if isinstance(data, list):
            row = data[0] if data else None
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            row = data["data"][0] if data["data"] else None
        elif isinstance(data, dict):
            row = data

        if not isinstance(row, dict):
            return None
        return self._extract_float(
            row,
            [
                "trade_price",
                "closing_price",
                "last",
                "price",
            ],
        )

    def get_accounts(self) -> list[dict[str, Any]]:
        """Fetch all account balances (private)."""
        result = self._private_request("GET", "/v1/accounts")
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and isinstance(result.get("data"), list):
            return result["data"]
        return []

    def get_balance(self, ticker: str) -> Any:
        """Fetch balance by asset symbol."""
        asset = self._asset_symbol(ticker)

        if self.private_api_enabled:
            accounts = self.get_accounts()
            for row in accounts:
                currency = str(row.get("currency", "")).upper()
                if currency == asset:
                    return row
            return None

        if self._private_client is not None:
            return self._private_client.get_balance(asset)

        raise RuntimeError("Private API disabled. Check credentials and PyJWT.")

    def get_available_krw(self) -> float | None:
        """Fetch available KRW balance for position sizing."""
        if self.private_api_enabled:
            for row in self.get_accounts():
                if str(row.get("currency", "")).upper() != "KRW":
                    continue
                value = self._extract_float(
                    row,
                    ["balance", "available_balance", "available", "free"],
                )
                if value is not None:
                    return value
            return None

        if self._private_client is not None:
            try:
                raw = self._private_client.get_balance("KRW")
                if isinstance(raw, (tuple, list)) and raw:
                    return float(raw[0])
                if isinstance(raw, dict):
                    return self._extract_float(raw, ["available_krw", "balance", "available"])
            except Exception:
                return None

        return None

    def get_total_asset_krw(self) -> float | None:
        """Estimate total account asset value in KRW."""
        if not self.private_api_enabled:
            return None

        total = 0.0
        try:
            accounts = self.get_accounts()
            krw_markets = set(self.get_market_codes(quote=self.quote_currency))
        except Exception:
            return None

        for row in accounts:
            currency = str(row.get("currency", "")).upper().strip()
            if not currency:
                continue
            balance = self._extract_float(row, ["balance"]) or 0.0
            locked = self._extract_float(row, ["locked"]) or 0.0
            amount = balance + locked
            if amount <= 0:
                continue

            if currency == self.quote_currency:
                total += amount
                continue

            market = f"{self.quote_currency}-{currency}"
            if market not in krw_markets:
                continue
            price = self.get_current_price(market)
            if price is None:
                continue
            total += amount * float(price)
        return total

    def get_order_chance(self, ticker: str) -> dict[str, Any] | None:
        """Fetch order chance info for a market."""
        market = self._normalize_market(ticker)
        result = self._private_request("GET", "/v1/orders/chance", params={"market": market})
        return result if isinstance(result, dict) else None

    def create_order(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Create order via v2 beta endpoint.

        Endpoint from docs: POST /v2/orders
        """
        if not self.enable_official_orders:
            raise RuntimeError("Official REST order execution is disabled.")
        result = self._private_request("POST", "/v2/orders", json_body=payload)
        return result if isinstance(result, dict) else None

    def get_order(self, uuid_value: str) -> dict[str, Any] | None:
        """Get order status by uuid."""
        result = self._private_request("GET", "/v1/order", params={"uuid": uuid_value})
        if isinstance(result, dict):
            if isinstance(result.get("data"), dict):
                return result["data"]
            return result
        return None

    def cancel_order(self, uuid_value: str) -> dict[str, Any] | None:
        """Cancel order via v2 beta endpoint."""
        if not self.enable_official_orders:
            raise RuntimeError("Official REST order execution is disabled.")
        result = self._private_request("DELETE", "/v2/order", params={"uuid": uuid_value})
        return result if isinstance(result, dict) else None

    def market_buy(self, ticker: str, quantity: float, price_krw: float | None = None) -> Any:
        """Place market buy order.

        If pybithumb is available, keep legacy behavior.
        Otherwise use REST beta endpoint with `ord_type=price` and KRW notional.
        """
        if self._private_client is not None:
            return self._private_client.buy_market_order(self._asset_symbol(ticker), quantity)

        if not self.enable_official_orders:
            raise RuntimeError(
                "Order execution disabled. Set enable_official_orders=True after validating params.",
            )

        market = self._normalize_market(ticker)
        notional = price_krw
        if notional is None:
            current = self.get_current_price(market)
            if current is None:
                raise RuntimeError("Cannot infer KRW notional from current price")
            notional = current * quantity

        payload = {
            "market": market,
            "side": "bid",
            "ord_type": "price",
            "price": self._format_number(notional),
        }
        return self.create_order(payload)

    def market_sell(self, ticker: str, quantity: float) -> Any:
        """Place market sell order."""
        if self._private_client is not None:
            return self._private_client.sell_market_order(self._asset_symbol(ticker), quantity)

        if not self.enable_official_orders:
            raise RuntimeError(
                "Order execution disabled. Set enable_official_orders=True after validating params.",
            )

        payload = {
            "market": self._normalize_market(ticker),
            "side": "ask",
            "ord_type": "market",
            "volume": self._format_number(quantity),
        }
        return self.create_order(payload)

    def execute_market_buy(
        self,
        ticker: str,
        quantity: float,
        price_krw: float | None = None,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 2.5,
        order_fill_poll_sec: float = 0.5,
        cancel_unfilled_before_retry: bool = True,
    ) -> Any:
        """Execute market buy with retry/cancel for unfilled orders."""
        if self._private_client is not None:
            return self.market_buy(ticker=ticker, quantity=quantity, price_krw=price_krw)

        attempts = max(1, int(order_retry_count))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                order = self.market_buy(ticker=ticker, quantity=quantity, price_krw=price_krw)
                uuid_value = self._extract_order_uuid(order)
                if not uuid_value:
                    return order

                if self._wait_for_order_done(
                    uuid_value=uuid_value,
                    wait_sec=order_fill_wait_sec,
                    poll_sec=order_fill_poll_sec,
                ):
                    return order

                if cancel_unfilled_before_retry:
                    try:
                        self.cancel_order(uuid_value)
                    except Exception as cancel_exc:
                        LOGGER.warning("Order cancel failed uuid=%s err=%s", uuid_value, cancel_exc)
                raise RuntimeError(f"Order not filled in time (uuid={uuid_value})")
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(order_retry_delay_sec)))

        raise RuntimeError(f"Market buy failed after retries: {last_exc}")

    def execute_market_sell(
        self,
        ticker: str,
        quantity: float,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 2.5,
        order_fill_poll_sec: float = 0.5,
        cancel_unfilled_before_retry: bool = True,
    ) -> Any:
        """Execute market sell with retry/cancel for unfilled orders."""
        if self._private_client is not None:
            return self.market_sell(ticker=ticker, quantity=quantity)

        attempts = max(1, int(order_retry_count))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                order = self.market_sell(ticker=ticker, quantity=quantity)
                uuid_value = self._extract_order_uuid(order)
                if not uuid_value:
                    return order

                if self._wait_for_order_done(
                    uuid_value=uuid_value,
                    wait_sec=order_fill_wait_sec,
                    poll_sec=order_fill_poll_sec,
                ):
                    return order

                if cancel_unfilled_before_retry:
                    try:
                        self.cancel_order(uuid_value)
                    except Exception as cancel_exc:
                        LOGGER.warning("Order cancel failed uuid=%s err=%s", uuid_value, cancel_exc)
                raise RuntimeError(f"Order not filled in time (uuid={uuid_value})")
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(order_retry_delay_sec)))

        raise RuntimeError(f"Market sell failed after retries: {last_exc}")

    def execute_entry(
        self,
        ticker: str,
        side: str,
        quantity: float,
        notional: float | None = None,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 2.5,
        order_fill_poll_sec: float = 0.5,
        cancel_unfilled_before_retry: bool = True,
    ) -> Any:
        side_norm = side.upper().strip()
        if side_norm != "LONG":
            raise RuntimeError("Bithumb spot supports LONG only")
        return self.execute_market_buy(
            ticker=ticker,
            quantity=quantity,
            price_krw=notional,
            order_retry_count=order_retry_count,
            order_retry_delay_sec=order_retry_delay_sec,
            order_fill_wait_sec=order_fill_wait_sec,
            order_fill_poll_sec=order_fill_poll_sec,
            cancel_unfilled_before_retry=cancel_unfilled_before_retry,
        )

    def execute_exit(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_retry_count: int = 2,
        order_retry_delay_sec: float = 0.8,
        order_fill_wait_sec: float = 2.5,
        order_fill_poll_sec: float = 0.5,
        cancel_unfilled_before_retry: bool = True,
    ) -> Any:
        _ = side
        return self.execute_market_sell(
            ticker=ticker,
            quantity=quantity,
            order_retry_count=order_retry_count,
            order_retry_delay_sec=order_retry_delay_sec,
            order_fill_wait_sec=order_fill_wait_sec,
            order_fill_poll_sec=order_fill_poll_sec,
            cancel_unfilled_before_retry=cancel_unfilled_before_retry,
        )

    def get_top_volume_tickers(self, limit: int = 5, quote: str = "KRW") -> list[str]:
        """Return top tickers by 24h quote volume from ticker snapshot."""
        markets = self.get_market_codes(quote=quote)
        if not markets:
            return []

        scored: list[tuple[float, str]] = []
        chunk_size = 30
        for idx in range(0, len(markets), chunk_size):
            chunk = markets[idx : idx + chunk_size]
            data = self._public_get("/v1/ticker", {"markets": ",".join(chunk)})
            rows: list[dict[str, Any]] = []

            if isinstance(data, list):
                rows = [row for row in data if isinstance(row, dict)]
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                rows = [row for row in data["data"] if isinstance(row, dict)]

            for row in rows:
                market = str(row.get("market", "")).upper()
                if not market:
                    continue
                volume_24h = self._extract_float(
                    row,
                    [
                        "acc_trade_price_24h",
                        "acc_trade_value_24h",
                        "quote_volume",
                        "volume_24h",
                    ],
                )
                if volume_24h is None:
                    continue
                scored.append((volume_24h, market))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ticker for _, ticker in scored[:limit]]

    def normalize_market(self, ticker: str) -> str:
        """Public helper for market code normalization."""
        return self._normalize_market(ticker)

    def get_api_usage_snapshot(self, reset: bool = False) -> dict[str, Any]:
        """Return API usage counters for public/private calls."""
        snapshot = deepcopy(self._api_usage)
        if reset:
            self._api_usage = {
                "public": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
                "private": {"total": 0, "success": 0, "fail": 0, "by_path": {}},
            }
        return snapshot

    def get_bithumb_notices(self, page: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch Bithumb notices.

        Official v2.1.5 docs endpoint:
        - GET /v1/notices
        """
        data = self._public_get("/v1/notices")
        rows: list[dict[str, Any]] = []
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        elif isinstance(data, dict):
            for key in ("data", "content", "list", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    rows = [row for row in value if isinstance(row, dict)]
                    break

        if not rows:
            return []

        start = max(0, (page - 1) * limit)
        end = start + max(1, limit)
        return rows[start:end]

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.public_api_enabled:
            return None

        def _fetch() -> Any:
            assert self._session is not None
            self._mark_api_usage(kind="public", path=path, success=False, before_call=True)
            response = self._session.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout_sec,
            )
            payload = self._decode_response(response)
            self._mark_api_usage(kind="public", path=path, success=True, before_call=False)
            return payload

        return self._retry_public(_fetch, f"GET {path}")

    def _private_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        if not self.private_api_enabled:
            raise RuntimeError("Private API disabled. Check requests/PyJWT/credentials.")

        assert self._session is not None
        assert jwt is not None

        query_dict = params if params else json_body if json_body else None
        token = self._build_jwt_token(query_dict)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        self._mark_api_usage(kind="private", path=path, success=False, before_call=True)
        response = self._session.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout_sec,
        )
        try:
            payload = self._decode_response(response)
            self._mark_api_usage(kind="private", path=path, success=True, before_call=False)
            return payload
        except Exception:
            self._mark_api_usage(kind="private", path=path, success=False, before_call=False)
            raise

    def _build_jwt_token(self, query_dict: dict[str, Any] | None = None) -> str:
        assert jwt is not None

        payload: dict[str, Any] = {
            "access_key": self.credentials.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": round(time.time() * 1000),
        }

        if query_dict:
            query_string = urlencode(query_dict, doseq=True)
            query_hash = hashlib.sha512(query_string.encode("utf-8")).hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        token = jwt.encode(payload, self.credentials.secret_key, algorithm="HS256")
        return token if isinstance(token, str) else token.decode("utf-8")

    def _decode_response(self, response) -> Any:
        try:
            payload = response.json()
        except Exception:
            response.raise_for_status()
            return None

        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {payload}")

        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            err = payload["error"]
            raise RuntimeError(f"API error: {err.get('name')} {err.get('message')}")

        return payload

    def _build_candle_endpoint(
        self,
        market: str,
        interval: str,
        count: int,
    ) -> tuple[str, dict[str, Any]]:
        interval = interval.lower().strip()
        params: dict[str, Any] = {"market": market, "count": count}

        if interval.startswith("minute"):
            unit = interval.replace("minute", "")
            if not unit.isdigit():
                unit = "5"
            return f"/v1/candles/minutes/{int(unit)}", params
        if interval in {"day", "days"}:
            return "/v1/candles/days", params
        if interval in {"week", "weeks"}:
            return "/v1/candles/weeks", params
        if interval in {"month", "months"}:
            return "/v1/candles/months", params

        return "/v1/candles/minutes/5", params

    def _candles_to_frame(self, data: Any):
        if pd is None:
            return None
        if not isinstance(data, list):
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            ts = (
                item.get("candle_date_time_kst")
                or item.get("candle_date_time_utc")
                or item.get("timestamp")
                or item.get("time")
            )
            rows.append(
                {
                    "time": ts,
                    "open": self._extract_float(item, ["opening_price", "open", "start"]),
                    "high": self._extract_float(item, ["high_price", "high", "max"]),
                    "low": self._extract_float(item, ["low_price", "low", "min"]),
                    "close": self._extract_float(item, ["trade_price", "closing_price", "close", "end"]),
                    "volume": self._extract_float(
                        item,
                        ["candle_acc_trade_volume", "units_traded", "volume"],
                    ),
                    "value": self._extract_float(
                        item,
                        ["candle_acc_trade_price", "acc_trade_price", "quote_volume"],
                    ),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame.dropna(subset=["time", "open", "high", "low", "close", "volume"])
        if frame.empty:
            return frame

        frame = frame.set_index("time").sort_index()
        return frame

    def _retry_public(self, fn, label: str):
        attempts = max(1, self.public_retry_count)
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                kind, path = self._extract_kind_path_from_label(label)
                self._mark_api_usage(kind=kind, path=path, success=False, before_call=False)
                LOGGER.warning(
                    "Public API failed (%s) attempt=%s/%s: %s",
                    label,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(self.public_retry_delay_sec)

        if last_exc is not None:
            LOGGER.warning("Public API exhausted retries (%s): %s", label, last_exc)
        return None

    def _normalize_market(self, ticker: str) -> str:
        value = ticker.upper().strip()
        if "-" in value:
            return value
        return f"{self.quote_currency}-{value}"

    def _mark_api_usage(self, kind: str, path: str, success: bool, before_call: bool) -> None:
        bucket = self._api_usage.get(kind)
        if not isinstance(bucket, dict):
            return
        by_path = bucket.setdefault("by_path", {})
        path_stat = by_path.setdefault(path, {"total": 0, "success": 0, "fail": 0})
        if before_call:
            bucket["total"] += 1
            path_stat["total"] += 1
            return
        if success:
            bucket["success"] += 1
            path_stat["success"] += 1
        else:
            bucket["fail"] += 1
            path_stat["fail"] += 1

    @staticmethod
    def _extract_kind_path_from_label(label: str) -> tuple[str, str]:
        if label.startswith("GET "):
            return "public", label.replace("GET ", "", 1).strip()
        return "public", label

    def _wait_for_order_done(
        self,
        uuid_value: str,
        wait_sec: float = 2.5,
        poll_sec: float = 0.5,
    ) -> bool:
        deadline = time.time() + max(0.0, float(wait_sec))
        while time.time() <= deadline:
            try:
                detail = self.get_order(uuid_value)
            except Exception as exc:
                LOGGER.warning("Order status fetch failed uuid=%s err=%s", uuid_value, exc)
                detail = None

            if isinstance(detail, dict) and self._is_order_done(detail):
                return True
            time.sleep(max(0.05, float(poll_sec)))
        return False

    @staticmethod
    def _extract_order_uuid(order: Any) -> str | None:
        if not isinstance(order, dict):
            return None
        uuid_value = str(order.get("uuid", "")).strip()
        return uuid_value or None

    @staticmethod
    def _is_order_done(order: dict[str, Any]) -> bool:
        state = str(order.get("state") or order.get("status") or "").lower().strip()
        remaining = BithumbExchange._extract_float(
            order,
            ["remaining_volume", "remaining", "unfilled_volume"],
        )
        if state in {"cancel", "cancelled", "canceled"}:
            return False
        if state in {"done", "completed", "filled"}:
            return remaining is None or remaining <= 0
        return bool(remaining is not None and remaining <= 0)

    @staticmethod
    def _asset_symbol(ticker: str) -> str:
        value = ticker.upper().strip()
        if "-" in value:
            return value.split("-", 1)[1]
        return value

    @staticmethod
    def _extract_float(row: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            if key not in row:
                continue
            try:
                value = row[key]
                if value is None:
                    continue
                return float(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _format_number(value: float) -> str:
        return format(float(value), ".12f").rstrip("0").rstrip(".")
