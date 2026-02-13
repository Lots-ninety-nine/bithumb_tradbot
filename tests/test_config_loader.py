from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from core.config_loader import load_bot_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_bot_config_with_defaults_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "exchange:",
                        "  provider: bybit",
                        "app:",
                        "  interval_sec: 30",
                        "bybit:",
                        "  enabled: true",
                        "  allow_short: true",
                        "  leverage: 3.0",
                        "news:",
                        "  enabled: true",
                        "  refresh_interval_sec: 120",
                        "notification:",
                        "  enabled: true",
                    ]
                ),
                encoding="utf-8",
            )

            os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.example/webhook"
            cfg = load_bot_config(str(cfg_path))

            self.assertEqual(cfg.app.interval_sec, 30)
            self.assertEqual(cfg.exchange.provider, "bybit")
            self.assertTrue(cfg.bybit.enabled)
            self.assertTrue(cfg.bybit.allow_short)
            self.assertEqual(cfg.bybit.leverage, 3.0)
            self.assertTrue(cfg.news.enabled)
            self.assertEqual(cfg.news.refresh_interval_sec, 120)
            self.assertTrue(cfg.notification.enabled)
            self.assertEqual(cfg.notification.discord_webhook_url, "https://discord.example/webhook")


if __name__ == "__main__":
    unittest.main()
