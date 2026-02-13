"""리스크 및 포지션 관리."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class Position:
    slot_id: int
    ticker: str
    quantity: float
    entry_price: float
    highest_price: float
    opened_at: datetime


@dataclass(slots=True)
class ExitDecision:
    should_exit: bool
    reason: str = ""
    trailing_stop_price: float | None = None


class RiskManager:
    """Seed / slot / stop rules manager."""

    def __init__(
        self,
        seed_krw: float = 150000.0,
        slot_count: int = 3,
        stop_loss_pct: float = 0.03,
        trailing_start_pct: float = 0.01,
        trailing_gap_pct: float = 0.015,
    ) -> None:
        self.seed_krw = seed_krw
        self.slot_count = slot_count
        self.stop_loss_pct = stop_loss_pct
        self.trailing_start_pct = trailing_start_pct
        self.trailing_gap_pct = trailing_gap_pct
        self.positions: dict[int, Position] = {}

    @property
    def slot_budget_krw(self) -> float:
        return self.seed_krw / self.slot_count

    def available_slots(self) -> list[int]:
        return [idx for idx in range(self.slot_count) if idx not in self.positions]

    def can_open(self, ticker: str) -> bool:
        if any(pos.ticker == ticker for pos in self.positions.values()):
            return False
        return bool(self.available_slots())

    def open_position(self, ticker: str, quantity: float, entry_price: float) -> Position | None:
        free = self.available_slots()
        if not free:
            return None
        slot_id = free[0]
        position = Position(
            slot_id=slot_id,
            ticker=ticker,
            quantity=quantity,
            entry_price=entry_price,
            highest_price=entry_price,
            opened_at=datetime.now(timezone.utc),
        )
        self.positions[slot_id] = position
        return position

    def close_position(self, slot_id: int) -> Position | None:
        return self.positions.pop(slot_id, None)

    def evaluate_exit(self, position: Position, current_price: float) -> ExitDecision:
        """Return exit decision for a single position."""
        if current_price <= 0:
            return ExitDecision(should_exit=False, reason="invalid_price")

        if current_price > position.highest_price:
            position.highest_price = current_price

        pnl_pct = (current_price - position.entry_price) / position.entry_price
        if pnl_pct <= -self.stop_loss_pct:
            return ExitDecision(should_exit=True, reason="stop_loss")

        if pnl_pct >= self.trailing_start_pct:
            trailing_stop = position.highest_price * (1 - self.trailing_gap_pct)
            if current_price <= trailing_stop:
                return ExitDecision(
                    should_exit=True,
                    reason="trailing_stop",
                    trailing_stop_price=trailing_stop,
                )
            return ExitDecision(
                should_exit=False,
                reason="trailing_active",
                trailing_stop_price=trailing_stop,
            )

        return ExitDecision(should_exit=False, reason="hold")
