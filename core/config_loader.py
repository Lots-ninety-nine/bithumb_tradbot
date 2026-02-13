"""Application config loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    dry_run: bool = True
    interval_sec: int = 60
    max_consecutive_errors: int = 5
    enable_official_orders: bool = False
    log_level: str = "INFO"


@dataclass(slots=True)
class ExchangeConfig:
    sample_ticker: str = "KRW-BTC"
    quote_currency: str = "KRW"
    public_retry_count: int = 2
    public_retry_delay_sec: float = 0.2
    timeout_sec: float = 8.0


@dataclass(slots=True)
class CollectorConfig:
    top_n: int = 5
    candle_count: int = 200
    intervals: list[str] = field(default_factory=lambda: ["minute1", "minute5", "minute15"])


@dataclass(slots=True)
class StrategyConfig:
    interval_priority: list[str] = field(default_factory=lambda: ["minute15", "minute5", "minute1"])
    min_data_rows: int = 35
    required_signal_count: int = 2
    rsi_buy_threshold: float = 30.0
    bollinger_touch_tolerance_pct: float = 0.0
    use_macd_golden_cross: bool = True


@dataclass(slots=True)
class LLMConfig:
    model_name: str = "gemini-1.5-flash"
    min_buy_confidence: float = 0.7
    max_dead_cat_risk: float = 0.55


@dataclass(slots=True)
class TradeConfig:
    seed_krw: float = 150000.0
    slot_count: int = 3
    min_order_krw: float = 5000.0
    max_spread_bps: float = 35.0


@dataclass(slots=True)
class RiskConfig:
    stop_loss_pct: float = 0.03
    trailing_start_pct: float = 0.01
    trailing_gap_pct: float = 0.015


@dataclass(slots=True)
class BotConfig:
    app: AppConfig = field(default_factory=AppConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bot_config(config_path: str = "config.yaml") -> BotConfig:
    """Load bot config from yaml file with defaults."""
    default = BotConfig()
    path = Path(config_path)
    if not path.exists():
        return default

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(default.to_dict(), raw)

    return BotConfig(
        app=AppConfig(**_only_known_keys(AppConfig, merged["app"])),
        exchange=ExchangeConfig(**_only_known_keys(ExchangeConfig, merged["exchange"])),
        collector=CollectorConfig(**_only_known_keys(CollectorConfig, merged["collector"])),
        strategy=StrategyConfig(**_only_known_keys(StrategyConfig, merged["strategy"])),
        llm=LLMConfig(**_only_known_keys(LLMConfig, merged["llm"])),
        trade=TradeConfig(**_only_known_keys(TradeConfig, merged["trade"])),
        risk=RiskConfig(**_only_known_keys(RiskConfig, merged["risk"])),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _only_known_keys(dataclass_type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = getattr(dataclass_type, "__dataclass_fields__", {})
    return {key: value for key, value in data.items() if key in allowed}
