"""Bybit AI Hybrid Agent orchestrator."""

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
from core.bybit_exchange import BybitExchange
from core.notifier import DiscordNotifier, NotifyEvent
from core.performance_tracker import PerformanceSnapshot, PerformanceTracker


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
        self.provider = "bybit"
        self.loop_interval_sec = int(config.app.interval_sec)
        self.dry_run = bool(config.app.dry_run)
        self.max_consecutive_errors = int(config.app.max_consecutive_errors)
        self.max_spread_bps = float(config.trade.max_spread_bps)
        self.min_order_notional = float(config.trade.min_order_notional)
        self.use_available_balance_as_seed = bool(config.trade.use_available_balance_as_seed)
        self.order_retry_count = int(config.trade.order_retry_count)
        self.order_retry_delay_sec = float(config.trade.order_retry_delay_sec)
        self.order_fill_wait_sec = float(config.trade.order_fill_wait_sec)
        self.order_fill_poll_sec = float(config.trade.order_fill_poll_sec)
        self.cancel_unfilled_before_retry = bool(config.trade.cancel_unfilled_before_retry)
        self.max_dead_cat_risk = float(config.llm.max_dead_cat_risk)
        self.allow_hold_buy = bool(config.llm.allow_hold_buy)
        self.hold_buy_min_confidence = float(config.llm.hold_buy_min_confidence)
        self.hold_buy_max_dead_cat_risk = float(config.llm.hold_buy_max_dead_cat_risk)
        self.hold_buy_min_advanced_score = float(config.llm.hold_buy_min_advanced_score)
        self.allow_long = bool(config.bybit.allow_long)
        self.allow_short = bool(config.bybit.allow_short)
        self.short_min_advanced_score = float(config.bybit.short_min_advanced_score)
        self.frame_priority = tuple(config.strategy.interval_priority)
        self.log_api_usage = bool(config.app.log_api_usage)
        self.log_performance = bool(config.app.log_performance)
        self.performance_log_interval_sec = int(config.app.performance_log_interval_sec)
        self._started_notified = False
        self._next_performance_log_at = 0.0
        self.position_sync_interval_sec = max(60, self.loop_interval_sec * 3)
        self._next_position_sync_at = 0.0

        from core.llm_analyzer import GeminiAnalyzer
        from core.risk_manager import RiskManager
        from core.advanced_signals import evaluate_advanced_signals
        from core.strategy import HardRuleConfig, TechnicalSignal, evaluate_hard_rule, evaluate_hard_rule_short
        from data.collector import MarketDataCollector
        from data.news_collector import MarketNewsCollector
        from data.rag_store import SimpleRAGStore

        self._technical_signal_cls = TechnicalSignal
        self._evaluate_hard_rule = evaluate_hard_rule
        self._evaluate_hard_rule_short = evaluate_hard_rule_short
        self._evaluate_advanced_signals = evaluate_advanced_signals
        self._rule_config = HardRuleConfig(
            min_data_rows=int(config.strategy.min_data_rows),
            required_signal_count=int(config.strategy.required_signal_count),
            rsi_buy_threshold=float(config.strategy.rsi_buy_threshold),
            rsi_sell_threshold=float(config.strategy.rsi_sell_threshold),
            bollinger_touch_tolerance_pct=float(config.strategy.bollinger_touch_tolerance_pct),
            use_macd_golden_cross=bool(config.strategy.use_macd_golden_cross),
            use_macd_dead_cross=bool(config.strategy.use_macd_dead_cross),
        )
        self._advanced_config = config.advanced

        self.exchange = build_exchange(config)
        self.asset_unit = str(getattr(self.exchange, "quote_currency", config.bybit.quote_coin)).upper()
        self.allow_short = self.allow_short and bool(getattr(self.exchange, "supports_short", False))
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
            min_sell_confidence=config.llm.min_sell_confidence,
        )
        self.risk = RiskManager(
            seed_krw=config.trade.seed_capital,
            slot_count=config.trade.slot_count,
            stop_loss_pct=config.risk.stop_loss_pct,
            trailing_start_pct=config.risk.trailing_start_pct,
            trailing_gap_pct=config.risk.trailing_gap_pct,
        )
        self.notifier = self._build_notifier(config=config)
        self.performance_tracker = PerformanceTracker(
            baseline_path=config.app.performance_baseline_path,
        )
        self._sync_remote_positions(force=True)

    def run_once(self) -> None:
        try:
            self._sync_remote_positions()
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

                long_signal = self._evaluate_hard_rule(frame, config=self._rule_config) if self.allow_long else self._technical_signal_cls(False, 0, ["long_disabled"], {})
                short_signal = (
                    self._evaluate_hard_rule_short(frame, config=self._rule_config)
                    if self.allow_short
                    else self._technical_signal_cls(False, 0, ["short_disabled"], {})
                )
                if not long_signal.is_buy_candidate and not short_signal.is_buy_candidate:
                    continue

                advanced = self._evaluate_advanced_signals(
                    frame=frame,
                    orderbook=snapshot.orderbook,
                    config=self._advanced_config,
                )
                candidate_sides: list[tuple[str, Any]] = []
                if long_signal.is_buy_candidate:
                    long_allowed = (not self._advanced_config.enabled) or advanced.buy_bias
                    if long_allowed:
                        candidate_sides.append(("LONG", long_signal))
                    else:
                        LOGGER.info(
                            "%s LONG advanced gate blocked score=%.3f reasons=%s",
                            ticker,
                            advanced.total_score,
                            advanced.reasons,
                        )
                if short_signal.is_buy_candidate:
                    short_allowed = (not self._advanced_config.enabled) or (
                        advanced.total_score <= -self.short_min_advanced_score
                    )
                    if short_allowed:
                        candidate_sides.append(("SHORT", short_signal))
                    else:
                        LOGGER.info(
                            "%s SHORT advanced gate blocked score=%.3f reasons=%s",
                            ticker,
                            advanced.total_score,
                            advanced.reasons,
                        )

                if not candidate_sides:
                    continue

                selected_side, signal = self._select_trade_side(
                    candidate_sides=candidate_sides,
                    advanced_score=advanced.total_score,
                )

                asset_symbol = self._asset_symbol(ticker)
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
                        "intended_side": selected_side,
                        "hard_rule": asdict(signal),
                        "hard_rule_long": asdict(long_signal),
                        "hard_rule_short": asdict(short_signal),
                        "advanced": advanced.to_dict(),
                    },
                    candle_payload=recent_candles,
                    rag_payload=news_context,
                )
                LOGGER.info(
                    "%s side=%s hard=%s advanced=%s llm=%s",
                    ticker,
                    selected_side,
                    asdict(signal),
                    advanced.to_dict(),
                    asdict(llm_decision),
                )

                allow_entry = False
                if selected_side == "LONG":
                    allow_entry = self.analyzer.allow_buy(
                        llm_decision,
                        max_dead_cat_risk=self.max_dead_cat_risk,
                    )
                    if (
                        not allow_entry
                        and self._allow_hold_buy_override(
                            decision=llm_decision,
                            advanced_score=advanced.total_score,
                        )
                    ):
                        allow_entry = True
                        LOGGER.info(
                            "%s LONG HOLD override accepted: conf=%.2f dead_cat=%.2f adv=%.3f",
                            ticker,
                            llm_decision.confidence,
                            llm_decision.dead_cat_bounce_risk
                            if llm_decision.dead_cat_bounce_risk is not None
                            else -1.0,
                            advanced.total_score,
                        )
                else:
                    allow_entry = self.analyzer.allow_sell(llm_decision)

                if not allow_entry:
                    LOGGER.info(
                        "%s %s LLM gate blocked decision=%s confidence=%.2f dead_cat=%.2f",
                        ticker,
                        selected_side,
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

                slot_budget_notional = self._resolve_slot_budget_notional()
                quantity = slot_budget_notional / current_price
                order_value_notional = quantity * current_price
                if order_value_notional < self.min_order_notional:
                    LOGGER.info(
                        "%s skipped by min order rule value=%.2f min=%.2f",
                        ticker,
                        order_value_notional,
                        self.min_order_notional,
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
                    side=selected_side,
                )
                if position is None:
                    continue

                if self.dry_run or not self.exchange.trading_enabled:
                    LOGGER.info(
                        "[DRY-RUN] ENTRY %s side=%s qty=%.8f price=%.2f slot=%s",
                        ticker,
                        selected_side,
                        quantity,
                        current_price,
                        position.slot_id,
                    )
                else:
                    if self.config.notification.notify_on_order_attempt:
                        self._notify(
                            NotifyEvent(
                                title="Order Attempt",
                                description=f"{ticker} {selected_side} 진입 주문 시도",
                                level="info",
                                fields=[
                                    DiscordNotifier.field("ticker", ticker, inline=True),
                                    DiscordNotifier.field("side", selected_side, inline=True),
                                    DiscordNotifier.field("qty", f"{quantity:.8f}", inline=True),
                                    DiscordNotifier.field("price", f"{current_price:.4f}", inline=True),
                                ],
                            )
                        )
                    try:
                        self.exchange.execute_entry(
                            ticker=ticker,
                            side=selected_side,
                            quantity=quantity,
                            notional=slot_budget_notional,
                            order_retry_count=self.order_retry_count,
                            order_retry_delay_sec=self.order_retry_delay_sec,
                            order_fill_wait_sec=self.order_fill_wait_sec,
                            order_fill_poll_sec=self.order_fill_poll_sec,
                            cancel_unfilled_before_retry=self.cancel_unfilled_before_retry,
                        )
                        LOGGER.info("ENTRY executed for %s side=%s", ticker, selected_side)
                        if self.config.notification.notify_on_order_success:
                            self._notify(
                                NotifyEvent(
                                    title="Order Filled",
                                    description=f"{ticker} {selected_side} 진입 체결",
                                    level="success",
                                    fields=[
                                        DiscordNotifier.field("ticker", ticker, inline=True),
                                        DiscordNotifier.field("side", selected_side, inline=True),
                                        DiscordNotifier.field("qty", f"{quantity:.8f}", inline=True),
                                        DiscordNotifier.field("price", f"{current_price:.4f}", inline=True),
                                    ],
                                )
                            )
                    except Exception as exc:
                        self.risk.close_position(position.slot_id)
                        LOGGER.warning("ENTRY failed for %s side=%s: %s", ticker, selected_side, exc)
                        if self.config.notification.notify_on_order_failure:
                            self._notify(
                                NotifyEvent(
                                    title="Order Failed",
                                    description=f"{ticker} {selected_side} 진입 주문 실패",
                                    level="error",
                                    fields=[
                                        DiscordNotifier.field("ticker", ticker, inline=True),
                                        DiscordNotifier.field("side", selected_side, inline=True),
                                        DiscordNotifier.field("qty", f"{quantity:.8f}", inline=True),
                                        DiscordNotifier.field("error", str(exc)[:900], inline=False),
                                    ],
                                )
                            )
                        continue

                if self.config.notification.notify_on_buy:
                    self._notify(
                        NotifyEvent(
                            title="ENTRY Signal",
                            description=f"{ticker} {selected_side} 진입 처리",
                            level="success",
                            fields=[
                                DiscordNotifier.field("ticker", ticker, inline=True),
                                DiscordNotifier.field("side", selected_side, inline=True),
                                DiscordNotifier.field("qty", f"{quantity:.8f}", inline=True),
                                DiscordNotifier.field("price", f"{current_price:.2f}", inline=True),
                                DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                            ],
                        )
                    )

            self._check_open_positions()
        finally:
            self.log_performance_summary_if_needed()
            self.log_api_usage_summary(reset=True)

    def _sync_remote_positions(self, force: bool = False) -> None:
        if self.dry_run:
            return
        if not hasattr(self.exchange, "get_open_positions"):
            return
        now_ts = time.time()
        if not force and now_ts < self._next_position_sync_at:
            return
        self._next_position_sync_at = now_ts + self.position_sync_interval_sec

        try:
            remote_positions = self.exchange.get_open_positions()
        except Exception as exc:
            LOGGER.warning("Failed to sync open positions: %s", exc)
            return
        if not isinstance(remote_positions, list):
            return

        remote_keys: set[tuple[str, str]] = set()
        for row in remote_positions:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("symbol", "")).upper().strip()
            qty = float(row.get("qty") or 0.0)
            entry_price = float(row.get("entry_price") or 0.0)
            side = str(row.get("side") or "LONG").upper().strip()
            if ticker == "" or qty <= 0 or entry_price <= 0:
                continue
            remote_keys.add((ticker, side))
            if any(pos.ticker == ticker for pos in self.risk.positions.values()):
                continue
            if not self.risk.can_open(ticker):
                LOGGER.warning(
                    "Open position exists on exchange but local slots full: %s side=%s qty=%.6f",
                    ticker,
                    side,
                    qty,
                )
                continue
            loaded = self.risk.open_position(
                ticker=ticker,
                quantity=qty,
                entry_price=entry_price,
                side=side,
            )
            if loaded is not None:
                LOGGER.info(
                    "Synced remote position slot=%s ticker=%s side=%s qty=%.6f entry=%.6f",
                    loaded.slot_id,
                    ticker,
                    loaded.side,
                    loaded.quantity,
                    loaded.entry_price,
                )

        to_remove: list[int] = []
        for slot_id, pos in self.risk.positions.items():
            if (pos.ticker, pos.side) not in remote_keys:
                to_remove.append(slot_id)
        for slot_id in to_remove:
            closed = self.risk.close_position(slot_id)
            if closed is not None:
                LOGGER.info(
                    "Local position dropped after remote sync: %s %s slot=%s",
                    closed.ticker,
                    closed.side,
                    slot_id,
                )

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
                    "[DRY-RUN] EXIT %s side=%s qty=%.8f reason=%s slot=%s",
                    position.ticker,
                    position.side,
                    position.quantity,
                    decision.reason,
                    slot_id,
                )
            else:
                if self.config.notification.notify_on_order_attempt:
                    self._notify(
                        NotifyEvent(
                            title="Order Attempt",
                            description=f"{position.ticker} {position.side} 청산 주문 시도",
                            level="info",
                            fields=[
                                DiscordNotifier.field("ticker", position.ticker, inline=True),
                                DiscordNotifier.field("side", position.side, inline=True),
                                DiscordNotifier.field("qty", f"{position.quantity:.8f}", inline=True),
                                DiscordNotifier.field("reason", decision.reason, inline=True),
                            ],
                        )
                    )
                try:
                    self.exchange.execute_exit(
                        ticker=position.ticker,
                        side=position.side,
                        quantity=position.quantity,
                        order_retry_count=self.order_retry_count,
                        order_retry_delay_sec=self.order_retry_delay_sec,
                        order_fill_wait_sec=self.order_fill_wait_sec,
                        order_fill_poll_sec=self.order_fill_poll_sec,
                        cancel_unfilled_before_retry=self.cancel_unfilled_before_retry,
                    )
                    LOGGER.info(
                        "EXIT executed for %s side=%s reason=%s",
                        position.ticker,
                        position.side,
                        decision.reason,
                    )
                    if self.config.notification.notify_on_order_success:
                        self._notify(
                            NotifyEvent(
                                title="Order Filled",
                                description=f"{position.ticker} {position.side} 청산 체결",
                                level="success",
                                fields=[
                                    DiscordNotifier.field("ticker", position.ticker, inline=True),
                                    DiscordNotifier.field("side", position.side, inline=True),
                                    DiscordNotifier.field("qty", f"{position.quantity:.8f}", inline=True),
                                    DiscordNotifier.field("reason", decision.reason, inline=True),
                                ],
                            )
                        )
                except Exception as exc:
                    LOGGER.warning("EXIT failed for %s side=%s: %s", position.ticker, position.side, exc)
                    if self.config.notification.notify_on_order_failure:
                        self._notify(
                            NotifyEvent(
                                title="Order Failed",
                                description=f"{position.ticker} {position.side} 청산 주문 실패",
                                level="error",
                                fields=[
                                    DiscordNotifier.field("ticker", position.ticker, inline=True),
                                    DiscordNotifier.field("side", position.side, inline=True),
                                    DiscordNotifier.field("qty", f"{position.quantity:.8f}", inline=True),
                                    DiscordNotifier.field("error", str(exc)[:900], inline=False),
                                ],
                            )
                        )
                    continue
            self.risk.close_position(slot_id)

            if self.config.notification.notify_on_sell:
                perf_fields = self._performance_fields()
                if position.side == "SHORT":
                    trade_pnl_krw = (position.entry_price - current_price) * position.quantity
                else:
                    trade_pnl_krw = (current_price - position.entry_price) * position.quantity
                trade_pnl_pct = (
                    (trade_pnl_krw / (position.entry_price * position.quantity)) * 100.0
                    if position.entry_price > 0
                    else 0.0
                )
                self._notify(
                    NotifyEvent(
                        title="SELL Exit",
                        description=f"{position.ticker} {position.side} 청산 처리",
                        level="warn" if decision.reason == "stop_loss" else "info",
                        fields=[
                            DiscordNotifier.field("ticker", position.ticker, inline=True),
                            DiscordNotifier.field("side", position.side, inline=True),
                            DiscordNotifier.field("qty", f"{position.quantity:.8f}", inline=True),
                            DiscordNotifier.field("reason", decision.reason, inline=True),
                            DiscordNotifier.field("entry_price", f"{position.entry_price:.2f}", inline=True),
                            DiscordNotifier.field("exit_price", f"{current_price:.2f}", inline=True),
                            DiscordNotifier.field(
                                "trade_pnl",
                                f"{trade_pnl_krw:+.4f}{self.asset_unit} ({trade_pnl_pct:+.2f}%)",
                                inline=False,
                            ),
                            DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                            *perf_fields,
                        ],
                    )
                )

    def run_forever(self) -> None:
        LOGGER.info("Trading loop started at %s", datetime.now(timezone.utc).isoformat())
        self.log_performance_summary_if_needed(force=True)
        if not self._started_notified and self.config.notification.notify_on_startup:
            perf_fields = self._performance_fields()
            self._notify(
                NotifyEvent(
                    title="Bot Started",
                    description="트레이딩 루프가 시작되었습니다.",
                    level="info",
                    fields=[
                        DiscordNotifier.field("provider", self.provider, inline=True),
                        DiscordNotifier.field("dry_run", self.dry_run, inline=True),
                        DiscordNotifier.field("interval_sec", self.loop_interval_sec, inline=True),
                        DiscordNotifier.field("slot_count", self.config.trade.slot_count, inline=True),
                        *perf_fields,
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

    def _allow_hold_buy_override(self, decision: Any, advanced_score: float) -> bool:
        if not self.allow_hold_buy:
            return False
        if str(getattr(decision, "decision", "")).upper().strip() != "HOLD":
            return False

        confidence = float(getattr(decision, "confidence", 0.0))
        if confidence < self.hold_buy_min_confidence:
            return False

        dead_cat = getattr(decision, "dead_cat_bounce_risk", None)
        if dead_cat is not None and float(dead_cat) > self.hold_buy_max_dead_cat_risk:
            return False

        if float(advanced_score) < self.hold_buy_min_advanced_score:
            return False
        return True

    @staticmethod
    def _select_trade_side(
        candidate_sides: list[tuple[str, Any]],
        advanced_score: float,
    ) -> tuple[str, Any]:
        if len(candidate_sides) == 1:
            return candidate_sides[0]

        by_side = {side: signal for side, signal in candidate_sides}
        if "LONG" in by_side and "SHORT" in by_side:
            if advanced_score > 0:
                return "LONG", by_side["LONG"]
            if advanced_score < 0:
                return "SHORT", by_side["SHORT"]

            long_score = int(getattr(by_side["LONG"], "score", 0))
            short_score = int(getattr(by_side["SHORT"], "score", 0))
            if short_score > long_score:
                return "SHORT", by_side["SHORT"]
            return "LONG", by_side["LONG"]

        return candidate_sides[0]

    def _resolve_slot_budget_notional(self) -> float:
        fallback = self.risk.slot_budget_krw
        if not self.use_available_balance_as_seed:
            return fallback
        if self.dry_run:
            return fallback

        if hasattr(self.exchange, "get_available_balance"):
            available_krw = self.exchange.get_available_balance()
        else:
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

    def log_performance_summary_if_needed(self, force: bool = False) -> None:
        if not self.log_performance:
            return
        now_ts = time.time()
        if not force and now_ts < self._next_performance_log_at:
            return
        self._next_performance_log_at = now_ts + self.performance_log_interval_sec

        snapshot = self._get_performance_snapshot()
        if snapshot is None:
            return
        LOGGER.info(
            "Performance since %s | baseline=%.4f%s current=%.4f%s pnl=%.4f%s (%.2f%%)",
            snapshot.started_at.isoformat(),
            snapshot.baseline_krw,
            self.asset_unit,
            snapshot.current_krw,
            self.asset_unit,
            snapshot.pnl_krw,
            self.asset_unit,
            snapshot.pnl_pct,
        )

    def _get_performance_snapshot(self) -> PerformanceSnapshot | None:
        if hasattr(self.exchange, "get_total_asset"):
            total_krw = self.exchange.get_total_asset()
        else:
            total_krw = self.exchange.get_total_asset_krw()
        if total_krw is None or total_krw <= 0:
            return None
        return self.performance_tracker.snapshot(current_krw=total_krw)

    def _performance_fields(self) -> list[dict[str, object]]:
        snapshot = self._get_performance_snapshot()
        if snapshot is None:
            return []
        return [
            DiscordNotifier.field("start_at", snapshot.started_at.isoformat(), inline=False),
            DiscordNotifier.field("start_asset", f"{snapshot.baseline_krw:.4f}{self.asset_unit}", inline=True),
            DiscordNotifier.field("current_asset", f"{snapshot.current_krw:.4f}{self.asset_unit}", inline=True),
            DiscordNotifier.field(
                "total_pnl",
                f"{snapshot.pnl_krw:+.4f}{self.asset_unit} ({snapshot.pnl_pct:+.2f}%)",
                inline=False,
            ),
        ]

    def _pick_frame(self, candles: dict[str, Any]):
        for interval in self.frame_priority:
            frame = candles.get(interval)
            if frame is not None and not frame.empty:
                return frame
        return None

    @staticmethod
    def _asset_symbol(ticker: str) -> str:
        value = ticker.upper().strip()
        if "-" in value:
            return value.split("-")[-1]
        for quote in ("USDT", "USDC", "KRW", "BTC", "ETH"):
            if value.endswith(quote) and len(value) > len(quote):
                return value[: -len(quote)]
        return value

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
        units = orderbook.get("orderbook_units")
        if isinstance(units, list) and units:
            top = units[0]
            if isinstance(top, dict):
                key = "bid_price" if side == "bid" else "ask_price"
                try:
                    return float(top.get(key))
                except Exception:
                    pass

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
    parser = argparse.ArgumentParser(description="Crypto AI Hybrid Agent")
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
    parser.add_argument(
        "--reset-performance",
        action="store_true",
        help="수익률 기준점(시작 자산)을 초기화하고 종료",
    )
    return parser.parse_args()


def build_exchange(config: BotConfig):
    return BybitExchange(
        base_url=config.bybit.base_url,
        category=config.bybit.category,
        quote_coin=config.bybit.quote_coin,
        account_type=config.bybit.account_type,
        recv_window=config.bybit.recv_window,
        position_idx=config.bybit.position_idx,
        leverage=config.bybit.leverage,
        public_retry_count=config.exchange.public_retry_count,
        public_retry_delay_sec=config.exchange.public_retry_delay_sec,
        timeout_sec=config.exchange.timeout_sec,
        enable_trading=not config.app.dry_run,
    )


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

    exchange = build_exchange(config)
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
        exchange = build_exchange(config)
        report = exchange.connectivity_report(sample_ticker=config.exchange.sample_ticker)
        tracker = PerformanceTracker(baseline_path=config.app.performance_baseline_path)
        try:
            if hasattr(exchange, "get_total_asset"):
                total_krw = exchange.get_total_asset()
            else:
                total_krw = exchange.get_total_asset_krw()
            if total_krw is not None and total_krw > 0:
                perf = tracker.snapshot(current_krw=total_krw)
                report["performance"] = {
                    "asset_unit": str(getattr(exchange, "quote_currency", config.bybit.quote_coin)).upper(),
                    "started_at": perf.started_at.isoformat(),
                    "baseline_krw": perf.baseline_krw,
                    "current_krw": perf.current_krw,
                    "pnl_krw": perf.pnl_krw,
                    "pnl_pct": perf.pnl_pct,
                }
        except Exception as exc:
            report["performance"] = None
            if not report.get("error"):
                report["error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    if args.validate_data:
        raise SystemExit(run_data_validation(config))

    if args.reset_performance:
        tracker = PerformanceTracker(baseline_path=config.app.performance_baseline_path)
        removed = tracker.reset()
        print(
            json.dumps(
                {
                    "ok": True,
                    "removed_existing_baseline": removed,
                    "baseline_path": config.app.performance_baseline_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

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
