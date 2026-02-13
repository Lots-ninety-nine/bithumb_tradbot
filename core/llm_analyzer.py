"""LLM analyzer (OpenAI default, Gemini optional)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai = None
    genai_types = None


@dataclass(slots=True)
class LLMDecision:
    decision: str
    reason: str
    confidence: float
    dead_cat_bounce_risk: float | None = None


class LLMAnalyzer:
    """Model-backed market context analyzer."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        min_buy_confidence: float = 0.7,
        min_sell_confidence: float = 0.7,
        openai_base_url: str = "https://api.openai.com/v1",
        openai_timeout_sec: float = 12.0,
    ) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        self.provider = str(provider).strip().lower()
        self.model_name = model_name
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_timeout_sec = float(openai_timeout_sec)
        self._openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._gemini_client = None

        if self.provider == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            if genai and gemini_key:
                self._gemini_client = genai.Client(api_key=gemini_key)

    @property
    def enabled(self) -> bool:
        if self.provider == "openai":
            return bool(self._openai_api_key)
        if self.provider == "gemini":
            return self._gemini_client is not None
        return False

    def analyze(
        self,
        ticker: str,
        technical_payload: dict[str, Any],
        candle_payload: list[dict[str, Any]],
        rag_payload: list[dict[str, Any]],
    ) -> LLMDecision:
        if self.provider == "openai":
            return self._analyze_openai(
                ticker=ticker,
                technical_payload=technical_payload,
                candle_payload=candle_payload,
                rag_payload=rag_payload,
            )
        if self.provider == "gemini":
            return self._analyze_gemini(
                ticker=ticker,
                technical_payload=technical_payload,
                candle_payload=candle_payload,
                rag_payload=rag_payload,
            )
        return LLMDecision(
            decision="HOLD",
            reason=f"Unsupported llm.provider={self.provider}",
            confidence=0.0,
            dead_cat_bounce_risk=None,
        )

    def _analyze_openai(
        self,
        ticker: str,
        technical_payload: dict[str, Any],
        candle_payload: list[dict[str, Any]],
        rag_payload: list[dict[str, Any]],
    ) -> LLMDecision:
        if not self._openai_api_key:
            return LLMDecision(
                decision="HOLD",
                reason="OPENAI_API_KEY not configured",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )

        prompt = self._build_prompt(
            ticker=ticker,
            technical_payload=technical_payload,
            candle_payload=candle_payload,
            rag_payload=rag_payload,
        )
        url = f"{self.openai_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a crypto trading risk analyst. "
                        "Return one JSON object only with keys: "
                        "decision, reason, confidence, dead_cat_bounce_risk."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=self.openai_timeout_sec,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            payload = resp.json()
            choices = payload.get("choices", []) if isinstance(payload, dict) else []
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"No choices in response: {payload}")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"Empty content: {payload}")
            return self._parse_json_response(content.strip())
        except Exception as exc:
            return LLMDecision(
                decision="HOLD",
                reason=f"OpenAI call failed: {exc}",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )

    def _analyze_gemini(
        self,
        ticker: str,
        technical_payload: dict[str, Any],
        candle_payload: list[dict[str, Any]],
        rag_payload: list[dict[str, Any]],
    ) -> LLMDecision:
        if self._gemini_client is None:
            return LLMDecision(
                decision="HOLD",
                reason="Gemini not configured",
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
            response = self._gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
                if genai_types is not None
                else None,
            )
            text = self._extract_gemini_text(response).strip()
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
            "입력 JSON을 분석해서 아래 형식의 JSON 한 줄만 출력하라.\n"
            '{"decision":"BUY|HOLD|SELL","reason":"...","confidence":0.0,"dead_cat_bounce_risk":0.0}\n'
            f"INPUT={json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text
        return ""

    def _parse_json_response(self, raw_text: str) -> LLMDecision:
        text = raw_text.strip().strip("`")
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

        try:
            data = json.loads(text)
            decision = str(data.get("decision", "HOLD")).upper().strip()
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
            return LLMDecision(
                decision="HOLD",
                reason=f"Unparseable LLM output: {raw_text[:160]}",
                confidence=0.0,
                dead_cat_bounce_risk=None,
            )

    @staticmethod
    def to_dict(decision: LLMDecision) -> dict[str, Any]:
        return asdict(decision)

    def allow_buy(self, decision: LLMDecision, max_dead_cat_risk: float = 0.55) -> bool:
        if decision.decision != "BUY":
            return False
        if decision.confidence < self.min_buy_confidence:
            return False
        if decision.dead_cat_bounce_risk is not None and decision.dead_cat_bounce_risk > max_dead_cat_risk:
            return False
        return True

    def allow_sell(self, decision: LLMDecision) -> bool:
        if decision.decision != "SELL":
            return False
        return decision.confidence >= self.min_sell_confidence


# Backward compatibility for old import path
GeminiAnalyzer = LLMAnalyzer
