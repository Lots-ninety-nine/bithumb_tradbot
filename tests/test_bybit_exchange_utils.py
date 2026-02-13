from __future__ import annotations

import hashlib
import hmac
import json
import unittest
from unittest.mock import patch

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

    def test_private_post_signature_uses_exact_sent_json(self) -> None:
        class _DummyResponse:
            status_code = 200

            @staticmethod
            def json() -> dict[str, object]:
                return {"retCode": 0, "retMsg": "OK", "result": {}}

        class _DummySession:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def request(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls.append(kwargs)
                return _DummyResponse()

        ex = BybitExchange(enable_trading=False)
        session = _DummySession()
        ex._session = session  # type: ignore[attr-defined]
        ex.credentials.api_key = "test_api_key"
        ex.credentials.secret_key = "test_secret_key"
        ex.recv_window = 5000

        body = {
            "category": "linear",
            "symbol": "ZROUSDT",
            "side": "Buy",
            "orderType": "Market",
            "qty": "14.1",
            "reduceOnly": False,
            "timeInForce": "IOC",
        }
        expected_body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)

        with patch("core.bybit_exchange.time.time", return_value=1700000000.123):
            out = ex._request("POST", "/v5/order/create", body=body, signed=True)  # type: ignore[attr-defined]

        self.assertEqual(out.get("retCode"), 0)
        self.assertEqual(len(session.calls), 1)
        call = session.calls[0]
        headers = call.get("headers")
        self.assertIsInstance(headers, dict)
        assert isinstance(headers, dict)
        self.assertEqual(call.get("data"), expected_body_json)
        self.assertEqual(headers.get("X-BAPI-TIMESTAMP"), "1700000000123")
        self.assertEqual(headers.get("X-BAPI-RECV-WINDOW"), "5000")

        sign_payload = f"1700000000123test_api_key5000{expected_body_json}"
        expected_sign = hmac.new(
            b"test_secret_key",
            sign_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(headers.get("X-BAPI-SIGN"), expected_sign)


if __name__ == "__main__":
    unittest.main()
