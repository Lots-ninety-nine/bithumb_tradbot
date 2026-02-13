"""Advanced market micro-signal evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from core.config_loader import AdvancedSignalConfig


@dataclass(slots=True)
class AdvancedSignalResult:
    total_score: float
    buy_bias: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_advanced_signals(
    frame: pd.DataFrame,
    orderbook: dict[str, Any] | None,
    config: AdvancedSignalConfig,
) -> AdvancedSignalResult:
    if frame is None or frame.empty or len(frame) < 30:
        return AdvancedSignalResult(
            total_score=0.0,
            buy_bias=False,
            reasons=["insufficient_data"],
            metrics={},
        )

    sr_score, sr_reason, sr_metrics = _support_resistance_score(
        frame=frame,
        support_proximity_pct=config.support_proximity_pct,
        resistance_proximity_pct=config.resistance_proximity_pct,
    )
    pattern_score, pattern_reason, pattern_metrics = _pattern_score(frame)
    ob_score, ob_reason, ob_metrics = _orderbook_score(
        orderbook=orderbook,
        imbalance_buy_threshold=config.imbalance_buy_threshold,
    )

    weighted = (
        sr_score * config.support_resistance_weight
        + pattern_score * config.pattern_weight
        + ob_score * config.orderbook_weight
    )
    reasons = [*sr_reason, *pattern_reason, *ob_reason]
    metrics = {
        "support_resistance": sr_metrics,
        "pattern": pattern_metrics,
        "orderbook": ob_metrics,
        "weighted_score": weighted,
    }
    return AdvancedSignalResult(
        total_score=float(weighted),
        buy_bias=weighted >= config.min_total_score,
        reasons=reasons,
        metrics=metrics,
    )


def _support_resistance_score(
    frame: pd.DataFrame,
    support_proximity_pct: float,
    resistance_proximity_pct: float,
) -> tuple[float, list[str], dict[str, Any]]:
    lookback = frame.tail(120).copy()
    close = float(lookback.iloc[-1]["close"])

    lows = lookback["low"]
    highs = lookback["high"]
    support = float(lows.nsmallest(5).median())
    resistance = float(highs.nlargest(5).median())

    near_support = close <= support * (1 + support_proximity_pct)
    near_resistance = close >= resistance * (1 - resistance_proximity_pct)
    breakout = close > resistance * (1 + resistance_proximity_pct)

    score = 0.0
    reasons: list[str] = []
    if near_support:
        score += 1.0
        reasons.append("near_support")
    if breakout:
        score += 0.7
        reasons.append("resistance_breakout")
    if near_resistance and not breakout:
        score -= 0.6
        reasons.append("near_resistance")

    metrics = {
        "close": close,
        "support": support,
        "resistance": resistance,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "breakout": breakout,
    }
    return score, reasons, metrics


def _pattern_score(frame: pd.DataFrame) -> tuple[float, list[str], dict[str, Any]]:
    last = frame.iloc[-1]
    prev = frame.iloc[-2]

    last_open = float(last["open"])
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])

    bullish_engulfing = (
        prev_close < prev_open
        and last_close > last_open
        and last_open <= prev_close
        and last_close >= prev_open
    )
    bearish_engulfing = (
        prev_close > prev_open
        and last_close < last_open
        and last_open >= prev_close
        and last_close <= prev_open
    )
    body = abs(last_close - last_open)
    lower_wick = min(last_open, last_close) - last_low
    upper_wick = last_high - max(last_open, last_close)
    hammer = lower_wick > body * 2 and upper_wick <= body * 0.6
    shooting_star = upper_wick > body * 2 and lower_wick <= body * 0.6

    score = 0.0
    reasons: list[str] = []
    if bullish_engulfing:
        score += 1.0
        reasons.append("bullish_engulfing")
    if hammer:
        score += 0.6
        reasons.append("hammer")
    if bearish_engulfing:
        score -= 1.0
        reasons.append("bearish_engulfing")
    if shooting_star:
        score -= 0.7
        reasons.append("shooting_star")

    metrics = {
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "hammer": hammer,
        "shooting_star": shooting_star,
    }
    return score, reasons, metrics


def _orderbook_score(
    orderbook: dict[str, Any] | None,
    imbalance_buy_threshold: float,
) -> tuple[float, list[str], dict[str, Any]]:
    if not orderbook:
        return 0.0, ["no_orderbook"], {}

    units = orderbook.get("orderbook_units")
    if not isinstance(units, list) or not units:
        return 0.0, ["no_orderbook_units"], {}

    top = units[:5]
    bid_sum = 0.0
    ask_sum = 0.0
    best_bid = None
    best_ask = None
    for row in top:
        if not isinstance(row, dict):
            continue
        bid_size = _to_float(row.get("bid_size"))
        ask_size = _to_float(row.get("ask_size"))
        ask_price = _to_float(row.get("ask_price"))
        bid_price = _to_float(row.get("bid_price"))
        bid_sum += bid_size
        ask_sum += ask_size
        if best_ask is None and ask_price > 0:
            best_ask = ask_price
        if best_bid is None and bid_price > 0:
            best_bid = bid_price

    denom = bid_sum + ask_sum
    imbalance = ((bid_sum - ask_sum) / denom) if denom > 0 else 0.0
    spread_bps = None
    if best_ask and best_bid:
        mid = (best_ask + best_bid) / 2
        if mid > 0:
            spread_bps = ((best_ask - best_bid) / mid) * 10000

    score = 0.0
    reasons: list[str] = []
    if imbalance >= imbalance_buy_threshold:
        score += 1.0
        reasons.append("bid_imbalance")
    elif imbalance <= -imbalance_buy_threshold:
        score -= 1.0
        reasons.append("ask_imbalance")

    if spread_bps is not None and spread_bps > 50:
        score -= 0.4
        reasons.append("wide_spread")

    metrics = {
        "imbalance": imbalance,
        "bid_sum": bid_sum,
        "ask_sum": ask_sum,
        "spread_bps": spread_bps,
    }
    return score, reasons, metrics


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0
