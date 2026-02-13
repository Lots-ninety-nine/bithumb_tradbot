"""Gemini API 분석기."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency import guard
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency import guard
    genai = None


@dataclass(slots=True)
class LLMDecision:
    """Normalized LLM output."""

    decision: str
    reason: str
    confidence: float
    dead_cat_bounce_risk: float | None = None


class GeminiAnalyzer:
    """Gemini based market context analyzer."""

    def __init__(self, model_name: str = "gemini-1.5-flash", min_buy_confidence: float = 0.7) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
        self.model_name = model_name
        self.min_buy_confidence = min_buy_confidence
        self._model = None

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if genai and api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name=model_name)

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def analyze(
        self,
        ticker: str,
        technical_payload: dict[str, Any],
        candle_payload: list[dict[str, Any]],
        rag_payload: list[dict[str, Any]],
    ) -> LLMDecision:
        """Call Gemini and parse JSON decision."""
        if not self.enabled:
            return LLMDecision(
                decision="HOLD",
                reason="Gemini not configured; fallback to HOLD",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )

        prompt = self._build_prompt(
            ticker=ticker,
            technical_payload=technical_payload,
            candle_payload=candle_payload,
            rag_payload=rag_payload,
        )

        try:
            response = self._model.generate_content(prompt)
            text = (response.text or "").strip()
            return self._parse_json_response(text)
        except Exception as exc:
            return LLMDecision(
                decision="HOLD",
                reason=f"Gemini call failed: {exc}",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )

    def _build_prompt(
        self,
        ticker: str,
        technical_payload: dict[str, Any],
        candle_payload: list[dict[str, Any]],
        rag_payload: list[dict[str, Any]],
    ) -> str:
        payload = {
            "ticker": ticker,
            "technical": technical_payload,
            "candles": candle_payload,
            "news_context": rag_payload,
        }
        return (
            "너는 한국 암호화폐 트레이딩 리스크 분석가다.\n"
            "아래 JSON을 읽고, 현재 반등이 데드캣 바운스일 가능성을 평가하라.\n"
            "반드시 JSON 한 줄만 출력하라: "
            '{"decision":"BUY|HOLD|SELL","reason":"...","confidence":0.0,'
            '"dead_cat_bounce_risk":0.0}\n'
            f"INPUT={json.dumps(payload, ensure_ascii=False)}"
        )

    def _parse_json_response(self, raw_text: str) -> LLMDecision:
        text = raw_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

        try:
            data = json.loads(text)
            decision = str(data.get("decision", "HOLD")).upper()
            reason = str(data.get("reason", "No reason returned"))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            dead_cat = data.get("dead_cat_bounce_risk")
            if dead_cat is not None:
                dead_cat = max(0.0, min(1.0, float(dead_cat)))
            if decision not in {"BUY", "HOLD", "SELL"}:
                decision = "HOLD"
            return LLMDecision(
                decision=decision,
                reason=reason,
                confidence=confidence,
                dead_cat_bounce_risk=dead_cat,
            )
        except Exception:
            fallback = LLMDecision(
                decision="HOLD",
                reason=f"Unparseable Gemini output: {raw_text[:160]}",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )
            return fallback

    @staticmethod
    def to_dict(decision: LLMDecision) -> dict[str, Any]:
        return asdict(decision)

    def allow_buy(self, decision: LLMDecision, max_dead_cat_risk: float = 0.55) -> bool:
        """BUY 실행 승인 게이트."""
        if decision.decision != "BUY":
            return False
        if decision.confidence < self.min_buy_confidence:
            return False
        if (
            decision.dead_cat_bounce_risk is not None
            and decision.dead_cat_bounce_risk > max_dead_cat_risk
        ):
            return False
        return True
