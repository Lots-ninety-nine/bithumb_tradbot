"""Notification utilities (Discord webhook)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class NotifyEvent:
    title: str
    description: str
    level: str = "info"
    fields: list[dict[str, str]] | None = None


class DiscordNotifier:
    """Small Discord webhook client with rate guard."""

    COLOR_MAP = {
        "info": 0x4E8BFF,
        "warn": 0xF1C40F,
        "error": 0xE74C3C,
        "success": 0x2ECC71,
    }

    def __init__(
        self,
        webhook_url: str,
        username: str = "BithumbTradBot",
        timeout_sec: float = 5.0,
        min_interval_sec: float = 1.0,
    ) -> None:
        self.webhook_url = webhook_url.strip()
        self.username = username
        self.timeout_sec = timeout_sec
        self.min_interval_sec = min_interval_sec
        self._last_sent_ts = 0.0
        self._session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, event: NotifyEvent) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        elapsed = now - self._last_sent_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

        payload = {
            "username": self.username,
            "embeds": [
                {
                    "title": event.title[:256],
                    "description": event.description[:4000],
                    "color": self.COLOR_MAP.get(event.level, self.COLOR_MAP["info"]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fields": (event.fields or [])[:25],
                }
            ],
        }
        try:
            response = self._session.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout_sec,
            )
            if response.status_code >= 400:
                LOGGER.warning("Discord webhook failed: %s %s", response.status_code, response.text)
                return False
            self._last_sent_ts = time.time()
            return True
        except Exception as exc:
            LOGGER.warning("Discord webhook error: %s", exc.__class__.__name__)
            return False

    @staticmethod
    def field(name: str, value: Any, inline: bool = False) -> dict[str, Any]:
        return {
            "name": str(name)[:256],
            "value": str(value)[:1024],
            "inline": inline,
        }
