"""Bithumb AI Hybrid Agent orchestrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import math
import time
from typing import Any

from core.exchange import BithumbExchange


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("tradbot")


class TradingOrchestrator:
    """Integrates Data/Intelligence/Risk agents."""

    def __init__(
        self,
        loop_interval_sec: int = 60,
        dry_run: bool = True,
        min_buy_confidence: float = 0.7,
        max_spread_bps: float = 35.0,
        min_order_krw: float = 5000.0,
        max_consecutive_errors: int = 5,
        enable_official_orders: bool = False,
    ) -> None:
        self.loop_interval_sec = loop_interval_sec
        self.dry_run = dry_run
        self.max_spread_bps = max_spread_bps
        self.min_order_krw = min_order_krw
        self.max_consecutive_errors = max_consecutive_errors

        from core.llm_analyzer import GeminiAnalyzer
        from core.risk_manager import RiskManager
        from core.strategy import evaluate_hard_rule
        from data.collector import MarketDataCollector
        from data.rag_store import SimpleRAGStore

        self._evaluate_hard_rule = evaluate_hard_rule
        self.exchange = BithumbExchange(enable_official_orders=enable_official_orders)
        self.collector = MarketDataCollector(exchange=self.exchange)
        self.rag_store = SimpleRAGStore()
        self.analyzer = GeminiAnalyzer(
            model_name="gemini-1.5-flash",
            min_buy_confidence=min_buy_confidence,
        )
        self.risk = RiskManager()

    def run_once(self) -> None:
        """Single cycle:
        1) Refresh watchlist / market data
        2) Hard-rule filter
        3) LLM check
        4) Risk checks and execution
        """
        if not self.collector.watchlist:
            watchlist = self.collector.refresh_watchlist()
            LOGGER.info("Watchlist updated: %s", watchlist)

        snapshots = self.collector.collect_once()
        if not snapshots:
            LOGGER.info("No market data collected. Skip this cycle.")
            return

        for ticker, snapshot in snapshots.items():
            frame = snapshot.candles.get("minute5")
            if frame is None or frame.empty:
                frame = snapshot.candles.get("minute1")
            if frame is None or frame.empty:
                continue

            signal = self._evaluate_hard_rule(frame)
            if not signal.is_buy_candidate:
                continue

            asset_symbol = ticker.split("-")[-1] if "-" in ticker else ticker
            news_context = self.rag_store.query_for_trade(ticker=asset_symbol, limit=3)
            recent_frame = frame.tail(40).reset_index()
            if len(recent_frame.columns) > 0:
                recent_frame = recent_frame.rename(columns={recent_frame.columns[0]: "timestamp"})
            if "timestamp" in recent_frame.columns:
                recent_frame["timestamp"] = recent_frame["timestamp"].astype(str)
            recent_candles = recent_frame.to_dict(orient="records")
            llm_decision = self.analyzer.analyze(
                ticker=ticker,
                technical_payload=asdict(signal),
                candle_payload=recent_candles,
                rag_payload=news_context,
            )
            LOGGER.info("%s signal=%s llm=%s", ticker, asdict(signal), asdict(llm_decision))

            if not self.analyzer.allow_buy(llm_decision):
                LOGGER.info(
                    "%s LLM buy gate blocked decision=%s confidence=%.2f dead_cat=%.2f",
                    ticker,
                    llm_decision.decision,
                    llm_decision.confidence,
                    llm_decision.dead_cat_bounce_risk
                    if llm_decision.dead_cat_bounce_risk is not None
                    else -1.0,
                )
                continue
            if not self.risk.can_open(ticker):
                continue

            current_price = float(frame.iloc[-1]["close"])
            if current_price <= 0:
                continue
            quantity = self.risk.slot_budget_krw / current_price
            order_value_krw = quantity * current_price
            if order_value_krw < self.min_order_krw:
                LOGGER.info(
                    "%s skipped by min order rule value=%.2f min=%.2f",
                    ticker,
                    order_value_krw,
                    self.min_order_krw,
                )
                continue

            spread_bps = self._calc_spread_bps(snapshot.orderbook)
            if spread_bps is not None and spread_bps > self.max_spread_bps:
                LOGGER.info(
                    "%s skipped by spread rule spread_bps=%.2f max=%.2f",
                    ticker,
                    spread_bps,
                    self.max_spread_bps,
                )
                continue

            position = self.risk.open_position(
                ticker=ticker,
                quantity=quantity,
                entry_price=current_price,
            )
            if position is None:
                continue

            if self.dry_run or not self.exchange.trading_enabled:
                LOGGER.info(
                    "[DRY-RUN] BUY %s qty=%.8f price=%.2f slot=%s",
                    ticker,
                    quantity,
                    current_price,
                    position.slot_id,
                )
            else:
                self.exchange.market_buy(ticker=ticker, quantity=quantity)
                LOGGER.info("BUY executed for %s", ticker)

        self._check_open_positions()

    def _check_open_positions(self) -> None:
        for slot_id, position in list(self.risk.positions.items()):
            current_price = self.exchange.get_current_price(position.ticker)
            if not current_price:
                continue
            decision = self.risk.evaluate_exit(position=position, current_price=current_price)
            if not decision.should_exit:
                continue

            if self.dry_run or not self.exchange.trading_enabled:
                LOGGER.info(
                    "[DRY-RUN] SELL %s qty=%.8f reason=%s slot=%s",
                    position.ticker,
                    position.quantity,
                    decision.reason,
                    slot_id,
                )
            else:
                self.exchange.market_sell(ticker=position.ticker, quantity=position.quantity)
                LOGGER.info("SELL executed for %s reason=%s", position.ticker, decision.reason)
            self.risk.close_position(slot_id)

    def run_forever(self) -> None:
        LOGGER.info("Trading loop started at %s", datetime.utcnow().isoformat())
        consecutive_errors = 0
        while True:
            try:
                self.run_once()
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                LOGGER.exception("Unhandled error in trading loop")
                if consecutive_errors >= self.max_consecutive_errors:
                    LOGGER.error(
                        "Consecutive error limit reached (%s). Cooling down for 5 minutes.",
                        consecutive_errors,
                    )
                    time.sleep(300)
                    consecutive_errors = 0
                    continue
                backoff = min(300, self.loop_interval_sec * math.pow(2, consecutive_errors))
                LOGGER.info("Backoff sleep %.0f sec after error", backoff)
                time.sleep(backoff)
                continue
            time.sleep(self.loop_interval_sec)

    @staticmethod
    def _calc_spread_bps(orderbook: dict[str, Any] | None) -> float | None:
        if not orderbook:
            return None

        bid_price = TradingOrchestrator._extract_best_price(orderbook, side="bid")
        ask_price = TradingOrchestrator._extract_best_price(orderbook, side="ask")
        if bid_price is None or ask_price is None:
            return None
        mid = (bid_price + ask_price) / 2.0
        if mid <= 0:
            return None
        return ((ask_price - bid_price) / mid) * 10000

    @staticmethod
    def _extract_best_price(orderbook: dict[str, Any], side: str) -> float | None:
        side_key_candidates = {
            "bid": ["bids", "bid", "bid_price"],
            "ask": ["asks", "ask", "ask_price"],
        }[side]
        for key in side_key_candidates:
            value = orderbook.get(key)
            if value is None:
                continue

            if isinstance(value, list) and value:
                top = value[0]
                if isinstance(top, dict):
                    for price_key in ("price", "ask_price", "bid_price"):
                        if price_key in top:
                            try:
                                return float(top[price_key])
                            except Exception:
                                continue
            elif isinstance(value, dict):
                for price_key in ("price", "ask_price", "bid_price"):
                    if price_key in value:
                        try:
                            return float(value[price_key])
                        except Exception:
                            continue
            else:
                try:
                    return float(value)
                except Exception:
                    continue
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bithumb AI Hybrid Agent")
    parser.add_argument(
        "--check-account",
        action="store_true",
        help="Run Step 1 account connectivity check and exit.",
    )
    parser.add_argument(
        "--sample-ticker",
        default="BTC",
        help="Ticker symbol for account check (default: BTC).",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run one trading cycle and exit.",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Run Step 2 data/indicator validation and exit.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Loop interval seconds (default: 60).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real order execution (default: dry-run).",
    )
    parser.add_argument(
        "--min-buy-confidence",
        type=float,
        default=0.7,
        help="Minimum Gemini confidence required for BUY execution.",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=35.0,
        help="Max spread in bps allowed for entry.",
    )
    parser.add_argument(
        "--min-order-krw",
        type=float,
        default=5000.0,
        help="Minimum order notional in KRW.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Loop cooldown trigger threshold for consecutive exceptions.",
    )
    parser.add_argument(
        "--enable-official-orders",
        action="store_true",
        help="Allow REST order execution (v2 beta endpoints). Use only after live verification.",
    )
    return parser.parse_args()


def run_data_validation() -> int:
    try:
        from core.strategy import indicator_health
        from data.collector import MarketDataCollector
    except ModuleNotFoundError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"Missing dependency: {exc.name}",
                    "hint": "Install requirements first: pip install -r requirements.txt",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    exchange = BithumbExchange()
    collector = MarketDataCollector(exchange=exchange, intervals=("minute1", "minute5", "minute15"))
    watchlist = collector.refresh_watchlist()
    snapshots = collector.collect_once()
    quality = collector.get_data_quality_report(min_rows=60)

    indicator_report: dict[str, dict[str, object]] = {}
    for ticker, snapshot in snapshots.items():
        frame = snapshot.candles.get("minute15")
        if frame is None or frame.empty:
            frame = snapshot.candles.get("minute5")
        if frame is None or frame.empty:
            frame = snapshot.candles.get("minute1")
        if frame is None or frame.empty:
            indicator_report[ticker] = {"ready": False, "reason": "no_candle_data"}
            continue
        indicator_report[ticker] = indicator_health(frame)

    print(
        json.dumps(
            {
                "ok": True,
                "watchlist": watchlist,
                "quality": quality,
                "indicator_report": indicator_report,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


def main() -> None:
    args = parse_args()

    if args.check_account:
        exchange = BithumbExchange()
        report = exchange.connectivity_report(sample_ticker=args.sample_ticker.upper())
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    if args.validate_data:
        raise SystemExit(run_data_validation())

    orchestrator = TradingOrchestrator(
        loop_interval_sec=args.interval,
        dry_run=not args.live,
        min_buy_confidence=args.min_buy_confidence,
        max_spread_bps=args.max_spread_bps,
        min_order_krw=args.min_order_krw,
        max_consecutive_errors=args.max_consecutive_errors,
        enable_official_orders=args.enable_official_orders,
    )
    if args.run_once:
        orchestrator.run_once()
        return
    orchestrator.run_forever()


if __name__ == "__main__":
    main()
