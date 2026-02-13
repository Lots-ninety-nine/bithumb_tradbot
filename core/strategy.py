"""기술적 지표 계산 및 Hard Rule 신호 평가."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class TechnicalSignal:
    """Hard rule evaluation result."""

    is_buy_candidate: bool
    score: int
    reasons: list[str] = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HardRuleConfig:
    """Configurable hard-rule thresholds."""

    min_data_rows: int = 35
    required_signal_count: int = 2
    rsi_buy_threshold: float = 30.0
    bollinger_touch_tolerance_pct: float = 0.0
    use_macd_golden_cross: bool = True


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI from close price."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Append indicators to OHLCV dataframe."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()

    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["bb_mid"] = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std20
    df["bb_lower"] = df["bb_mid"] - 2 * std20

    df["rsi14"] = calculate_rsi(df["close"], period=14)
    return df


def evaluate_hard_rule(frame: pd.DataFrame, config: HardRuleConfig | None = None) -> TechnicalSignal:
    """Evaluate technical buy conditions.

    Buy candidate if at least 2/3 conditions are met:
    - RSI <= 30
    - Close touches lower Bollinger band
    - MACD golden cross
    """
    rule = config or HardRuleConfig()
    df = add_indicators(frame)
    if df.empty or len(df) < rule.min_data_rows:
        return TechnicalSignal(
            is_buy_candidate=False,
            score=0,
            reasons=["insufficient_data"],
        )

    last = df.iloc[-1]
    prev = df.iloc[-2]

    cond_rsi = pd.notna(last["rsi14"]) and float(last["rsi14"]) <= rule.rsi_buy_threshold
    lower_touch_price = None
    if pd.notna(last["bb_lower"]):
        lower_touch_price = float(last["bb_lower"]) * (1 + rule.bollinger_touch_tolerance_pct)
    cond_bb = lower_touch_price is not None and float(last["close"]) <= lower_touch_price
    cond_macd = (
        pd.notna(prev["macd"])
        and pd.notna(prev["macd_signal"])
        and pd.notna(last["macd"])
        and pd.notna(last["macd_signal"])
        and float(prev["macd"]) <= float(prev["macd_signal"])
        and float(last["macd"]) > float(last["macd_signal"])
    ) if rule.use_macd_golden_cross else False

    reasons: list[str] = []
    if cond_rsi:
        reasons.append("rsi_oversold")
    if cond_bb:
        reasons.append("bollinger_lower_touch")
    if cond_macd:
        reasons.append("macd_golden_cross")

    score = int(cond_rsi) + int(cond_bb) + int(cond_macd)
    return TechnicalSignal(
        is_buy_candidate=score >= rule.required_signal_count,
        score=score,
        reasons=reasons,
        indicators={
            "close": float(last["close"]),
            "rsi14": float(last["rsi14"]) if pd.notna(last["rsi14"]) else None,
            "ma20": float(last["ma20"]) if pd.notna(last["ma20"]) else None,
            "ma60": float(last["ma60"]) if pd.notna(last["ma60"]) else None,
            "bb_upper": float(last["bb_upper"]) if pd.notna(last["bb_upper"]) else None,
            "bb_lower": float(last["bb_lower"]) if pd.notna(last["bb_lower"]) else None,
            "macd": float(last["macd"]) if pd.notna(last["macd"]) else None,
            "macd_signal": float(last["macd_signal"]) if pd.notna(last["macd_signal"]) else None,
        },
    )


def indicator_health(frame: pd.DataFrame) -> dict[str, Any]:
    """Return readiness/validity summary for major indicators."""
    df = add_indicators(frame)
    if df.empty:
        return {
            "ready": False,
            "rows": 0,
            "non_null_latest": {},
            "nan_ratio": {},
        }

    latest = df.iloc[-1]
    keys = ["rsi14", "ma20", "ma60", "bb_upper", "bb_lower", "macd", "macd_signal"]
    non_null_latest = {key: bool(pd.notna(latest[key])) for key in keys}
    nan_ratio = {key: float(df[key].isna().mean()) for key in keys}

    return {
        "ready": all(non_null_latest.values()),
        "rows": int(len(df)),
        "non_null_latest": non_null_latest,
        "nan_ratio": nan_ratio,
    }
