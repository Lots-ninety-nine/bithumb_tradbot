"""Portfolio performance tracker persisted to disk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(slots=True)
class PerformanceSnapshot:
    started_at: datetime
    baseline_krw: float
    current_krw: float
    pnl_krw: float
    pnl_pct: float


class PerformanceTracker:
    """Track PnL percentage from baseline account value."""

    def __init__(self, baseline_path: str = "data/performance_baseline.json") -> None:
        self.path = Path(baseline_path)

    def snapshot(self, current_krw: float, now: datetime | None = None) -> PerformanceSnapshot:
        now_dt = now or datetime.now(timezone.utc)
        started_at, baseline = self._load_or_init_baseline(current_krw=current_krw, now=now_dt)
        pnl_krw = current_krw - baseline
        pnl_pct = (pnl_krw / baseline * 100.0) if baseline > 0 else 0.0
        return PerformanceSnapshot(
            started_at=started_at,
            baseline_krw=baseline,
            current_krw=current_krw,
            pnl_krw=pnl_krw,
            pnl_pct=pnl_pct,
        )

    def reset(self) -> bool:
        if not self.path.exists():
            return False
        self.path.unlink()
        return True

    def _load_or_init_baseline(self, current_krw: float, now: datetime) -> tuple[datetime, float]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                started_at = datetime.fromisoformat(str(data.get("started_at")))
                baseline = float(data.get("baseline_krw"))
                if baseline > 0:
                    return self._as_utc(started_at), baseline
            except Exception:
                pass

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": self._as_utc(now).isoformat(),
            "baseline_krw": float(current_krw),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self._as_utc(now), float(current_krw)

    @staticmethod
    def _as_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

