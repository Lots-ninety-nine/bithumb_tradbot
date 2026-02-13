"""Bithumb AI Hybrid Agent orchestrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from core.config_loader import BotConfig, load_bot_config
from core.exchange import BithumbExchange
from core.notifier import DiscordNotifier, NotifyEvent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("tradbot")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")


class TradingOrchestrator:
    """Integrates Data/Intelligence/Risk agents."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.loop_interval_sec = int(config.app.interval_sec)
        self.dry_run = bool(config.app.dry_run)
        self.max_consecutive_errors = int(config.app.max_consecutive_errors)
        self.max_spread_bps = float(config.trade.max_spread_bps)
        self.min_order_krw = float(config.trade.min_order_krw)
        self.use_available_krw_as_seed = bool(config.trade.use_available_krw_as_seed)
        self.order_retry_count = int(config.trade.order_retry_count)
        self.order_retry_delay_sec = float(config.trade.order_retry_delay_sec)
        self.order_fill_wait_sec = float(config.trade.order_fill_wait_sec)
        self.order_fill_poll_sec = float(config.trade.order_fill_poll_sec)
        self.cancel_unfilled_before_retry = bool(config.trade.cancel_unfilled_before_retry)
        self.max_dead_cat_risk = float(config.llm.max_dead_cat_risk)
        self.frame_priority = tuple(config.strategy.interval_priority)
        self.log_api_usage = bool(config.app.log_api_usage)
        self._started_notified = False

        from core.llm_analyzer import GeminiAnalyzer
        from core.risk_manager import RiskManager
        from core.advanced_signals import evaluate_advanced_signals
        from core.strategy import HardRuleConfig, evaluate_hard_rule
        from data.collector import MarketDataCollector
        from data.news_collector import MarketNewsCollector
        from data.rag_store import SimpleRAGStore

        self._evaluate_hard_rule = evaluate_hard_rule
        self._evaluate_advanced_signals = evaluate_advanced_signals
        self._rule_config = HardRuleConfig(
            min_data_rows=int(config.strategy.min_data_rows),
            required_signal_count=int(config.strategy.required_signal_count),
            rsi_buy_threshold=float(config.strategy.rsi_buy_threshold),
            bollinger_touch_tolerance_pct=float(config.strategy.bollinger_touch_tolerance_pct),
            use_macd_golden_cross=bool(config.strategy.use_macd_golden_cross),
        )
        self._advanced_config = config.advanced

        self.exchange = BithumbExchange(
            quote_currency=config.exchange.quote_currency,
            public_retry_count=config.exchange.public_retry_count,
            public_retry_delay_sec=config.exchange.public_retry_delay_sec,
            timeout_sec=config.exchange.timeout_sec,
            enable_official_orders=config.app.enable_official_orders,
        )
        self.collector = MarketDataCollector(
            exchange=self.exchange,
            intervals=tuple(config.collector.intervals),
            top_n=config.collector.top_n,
            max_watchlist=config.collector.max_watchlist,
            candle_count=config.collector.candle_count,
            extra_watchlist=tuple(config.collector.extra_watchlist),
        )
        self.rag_store = SimpleRAGStore()
        self.news_collector = MarketNewsCollector(
            exchange=self.exchange,
            rag_store=self.rag_store,
            config=config.news,
        )
        self.news_refresh_interval_sec = int(config.news.refresh_interval_sec)
        self._next_news_refresh_at = 0.0
        self.analyzer = GeminiAnalyzer(
            model_name=config.llm.model_name,
            min_buy_confidence=config.llm.min_buy_confidence,
        )
        self.risk = RiskManager(
            seed_krw=config.trade.seed_krw,
            slot_count=config.trade.slot_count,
            stop_loss_pct=config.risk.stop_loss_pct,
            trailing_start_pct=config.risk.trailing_start_pct,
            trailing_gap_pct=config.risk.trailing_gap_pct,
        )
        self.notifier = self._build_notifier(config=config)

    def run_once(self) -> None:
        try:
            if not self.collector.watchlist:
                watchlist = self.collector.refresh_watchlist()
                LOGGER.info("Watchlist updated: %s", watchlist)
            self._refresh_news_if_needed()

            snapshots = self.collector.collect_once()
            if not snapshots:
                LOGGER.info("No market data collected. Skip this cycle.")
                return

            for ticker, snapshot in snapshots.items():
                frame = self._pick_frame(snapshot.candles)
                if frame is None or frame.empty:
                    continue

                signal = self._evaluate_hard_rule(frame, config=self._rule_config)
                if not signal.is_buy_candidate:
                    continue

                advanced = self._evaluate_advanced_signals(
                    frame=frame,
                    orderbook=snapshot.orderbook,
                    config=self._advanced_config,
                )
                if self._advanced_config.enabled and not advanced.buy_bias:
                    LOGGER.info(
                        "%s advanced gate blocked score=%.3f reasons=%s",
                        ticker,
                        advanced.total_score,
                        advanced.reasons,
                    )
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
                    technical_payload={
                        "hard_rule": asdict(signal),
                        "advanced": advanced.to_dict(),
                    },
                    candle_payload=recent_candles,
                    rag_payload=news_context,
                )
                LOGGER.info(
                    "%s hard=%s advanced=%s llm=%s",
                    ticker,
                    asdict(signal),
                    advanced.to_dict(),
                    asdict(llm_decision),
                )

                if not self.analyzer.allow_buy(llm_decision, max_dead_cat_risk=self.max_dead_cat_risk):
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

                slot_budget_krw = self._resolve_slot_budget_krw()
                quantity = slot_budget_krw / current_price
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
                    try:
                        self.exchange.execute_market_buy(
                            ticker=ticker,
                            quantity=quantity,
                            price_krw=slot_budget_krw,
                            order_retry_count=self.order_retry_count,
                            order_retry_delay_sec=self.order_retry_delay_sec,
                            order_fill_wait_sec=self.order_fill_wait_sec,
                            order_fill_poll_sec=self.order_fill_poll_sec,
                            cancel_unfilled_before_retry=self.cancel_unfilled_before_retry,
                        )
                        LOGGER.info("BUY executed for %s", ticker)
                    except Exception as exc:
                        self.risk.close_position(position.slot_id)
                        LOGGER.warning("BUY failed for %s: %s", ticker, exc)
                        continue

                if self.config.notification.notify_on_buy:
                    self._notify(
                        NotifyEvent(
                            title="BUY Signal",
                            description=f"{ticker} 진입 처리",
                            level="success",
                            fields=[
                                DiscordNotifier.field("ticker", ticker, inline=True),
                                DiscordNotifier.field("qty", f"{quantity:.8f}", inline=True),
                                DiscordNotifier.field("price", f"{current_price:.2f}", inline=True),
                                DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                            ],
                        )
                    )

            self._check_open_positions()
        finally:
            self.log_api_usage_summary(reset=True)

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
                try:
                    self.exchange.execute_market_sell(
                        ticker=position.ticker,
                        quantity=position.quantity,
                        order_retry_count=self.order_retry_count,
                        order_retry_delay_sec=self.order_retry_delay_sec,
                        order_fill_wait_sec=self.order_fill_wait_sec,
                        order_fill_poll_sec=self.order_fill_poll_sec,
                        cancel_unfilled_before_retry=self.cancel_unfilled_before_retry,
                    )
                    LOGGER.info("SELL executed for %s reason=%s", position.ticker, decision.reason)
                except Exception as exc:
                    LOGGER.warning("SELL failed for %s: %s", position.ticker, exc)
                    continue
            self.risk.close_position(slot_id)

            if self.config.notification.notify_on_sell:
                self._notify(
                    NotifyEvent(
                        title="SELL Exit",
                        description=f"{position.ticker} 청산 처리",
                        level="warn" if decision.reason == "stop_loss" else "info",
                        fields=[
                            DiscordNotifier.field("ticker", position.ticker, inline=True),
                            DiscordNotifier.field("qty", f"{position.quantity:.8f}", inline=True),
                            DiscordNotifier.field("reason", decision.reason, inline=True),
                            DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                        ],
                    )
                )

    def run_forever(self) -> None:
        LOGGER.info("Trading loop started at %s", datetime.now(timezone.utc).isoformat())
        if not self._started_notified and self.config.notification.notify_on_startup:
            self._notify(
                NotifyEvent(
                    title="Bot Started",
                    description="트레이딩 루프가 시작되었습니다.",
                    level="info",
                    fields=[
                        DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                        DiscordNotifier.field("interval_sec", self.loop_interval_sec, inline=True),
                        DiscordNotifier.field("slot_count", self.config.trade.slot_count, inline=True),
                    ],
                )
            )
            self._started_notified = True

        consecutive_errors = 0
        while True:
            try:
                self.run_once()
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                LOGGER.exception("Unhandled error in trading loop")
                if self.config.notification.notify_on_error:
                    self._notify(
                        NotifyEvent(
                            title="Bot Error",
                            description="트레이딩 루프 예외 발생",
                            level="error",
                            fields=[
                                DiscordNotifier.field(
                                    "consecutive_errors",
                                    consecutive_errors,
                                    inline=True,
                                )
                            ],
                        )
                    )
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

    def _refresh_news_if_needed(self) -> None:
        if not self.config.news.enabled:
            return
        now_ts = time.time()
        if now_ts < self._next_news_refresh_at:
            return

        result = self.news_collector.collect_once(self.collector.watchlist)
        self._next_news_refresh_at = now_ts + self.news_refresh_interval_sec
        LOGGER.info("News refreshed: %s", result)
        if self.config.notification.notify_on_news_refresh:
            self._notify(
                NotifyEvent(
                    title="News Refresh",
                    description="RAG 뉴스 컨텍스트 갱신",
                    level="info",
                    fields=[
                        DiscordNotifier.field("stored", result.get("stored", 0), inline=True),
                        DiscordNotifier.field("sources", result.get("source_counts", {}), inline=False),
                    ],
                )
            )

    def _build_notifier(self, config: BotConfig) -> DiscordNotifier | None:
        if not config.notification.enabled:
            return None
        webhook = config.notification.discord_webhook_url
        if not webhook:
            LOGGER.warning("notification.enabled=true but discord_webhook_url is empty")
            return None
        return DiscordNotifier(
            webhook_url=webhook,
            username=config.notification.username,
            timeout_sec=config.notification.timeout_sec,
            min_interval_sec=config.notification.min_interval_sec,
        )

    def _notify(self, event: NotifyEvent) -> bool:
        if self.notifier is None:
            return False
        ok = self.notifier.send(event)
        if not ok:
            LOGGER.warning("Notification send failed: title=%s", event.title)
        return ok

    def _resolve_slot_budget_krw(self) -> float:
        fallback = self.risk.slot_budget_krw
        if not self.use_available_krw_as_seed:
            return fallback
        if self.dry_run:
            return fallback

        available_krw = self.exchange.get_available_krw()
        if available_krw is None or available_krw <= 0:
            return fallback

        free_slots = max(1, len(self.risk.available_slots()))
        return max(0.0, available_krw / free_slots)

    def log_api_usage_summary(self, reset: bool = True) -> None:
        if not self.log_api_usage:
            return
        usage = self.exchange.get_api_usage_snapshot(reset=reset)
        public = usage.get("public", {})
        private = usage.get("private", {})
        LOGGER.info(
            "API usage public(total=%s success=%s fail=%s) private(total=%s success=%s fail=%s)",
            public.get("total", 0),
            public.get("success", 0),
            public.get("fail", 0),
            private.get("total", 0),
            private.get("success", 0),
            private.get("fail", 0),
        )

        top_paths = sorted(
            (public.get("by_path") or {}).items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True,
        )[:6]
        if top_paths:
            compact = {
                path: {
                    "t": stat.get("total", 0),
                    "s": stat.get("success", 0),
                    "f": stat.get("fail", 0),
                }
                for path, stat in top_paths
            }
            LOGGER.info("API public by_path(top): %s", compact)

    def _pick_frame(self, candles: dict[str, Any]):
        for interval in self.frame_priority:
            frame = candles.get(interval)
            if frame is not None and not frame.empty:
                return frame
        return None

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
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 설정파일 경로 (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--check-account",
        action="store_true",
        help="계좌/API 연동 상태를 출력하고 종료",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="데이터/지표 품질 점검 후 종료",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="루프 1회 실행 후 종료",
    )
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="디스코드 웹훅 테스트 알림 전송 후 종료",
    )
    return parser.parse_args()


def run_data_validation(config: BotConfig) -> int:
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

    exchange = BithumbExchange(
        quote_currency=config.exchange.quote_currency,
        public_retry_count=config.exchange.public_retry_count,
        public_retry_delay_sec=config.exchange.public_retry_delay_sec,
        timeout_sec=config.exchange.timeout_sec,
        enable_official_orders=config.app.enable_official_orders,
    )
    collector = MarketDataCollector(
        exchange=exchange,
        intervals=tuple(config.collector.intervals),
        top_n=config.collector.top_n,
        max_watchlist=config.collector.max_watchlist,
        candle_count=config.collector.candle_count,
        extra_watchlist=tuple(config.collector.extra_watchlist),
    )

    watchlist = collector.refresh_watchlist()
    snapshots = collector.collect_once()
    quality = collector.get_data_quality_report(min_rows=max(30, config.strategy.min_data_rows))

    indicator_report: dict[str, dict[str, object]] = {}
    for ticker, snapshot in snapshots.items():
        frame = None
        for interval in config.strategy.interval_priority:
            cand = snapshot.candles.get(interval)
            if cand is not None and not cand.empty:
                frame = cand
                break

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
    config = load_bot_config(args.config)

    level_name = str(config.app.log_level).upper().strip()
    logging.getLogger().setLevel(getattr(logging, level_name, logging.INFO))

    if args.check_account:
        exchange = BithumbExchange(
            quote_currency=config.exchange.quote_currency,
            public_retry_count=config.exchange.public_retry_count,
            public_retry_delay_sec=config.exchange.public_retry_delay_sec,
            timeout_sec=config.exchange.timeout_sec,
            enable_official_orders=config.app.enable_official_orders,
        )
        report = exchange.connectivity_report(sample_ticker=config.exchange.sample_ticker)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    if args.validate_data:
        raise SystemExit(run_data_validation(config))

    if args.test_notify:
        notifier = DiscordNotifier(
            webhook_url=config.notification.discord_webhook_url,
            username=config.notification.username,
            timeout_sec=config.notification.timeout_sec,
            min_interval_sec=config.notification.min_interval_sec,
        )
        sent = notifier.send(
            NotifyEvent(
                title="Bot Notification Test",
                description="수동 테스트 알림입니다.",
                level="info",
            )
        )
        print(
            json.dumps(
                {
                    "notification_enabled": config.notification.enabled,
                    "webhook_configured": bool(config.notification.discord_webhook_url.strip()),
                    "sent": sent,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    orchestrator = TradingOrchestrator(config=config)
    if args.run_once:
        orchestrator.run_once()
        return
    orchestrator.run_forever()


if __name__ == "__main__":
    main()
