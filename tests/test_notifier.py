from __future__ import annotations

import unittest
from unittest.mock import Mock

from core.notifier import DiscordNotifier, NotifyEvent


class NotifierTest(unittest.TestCase):
    def test_send_with_mocked_session(self) -> None:
        notifier = DiscordNotifier(
            webhook_url="https://discord.example/webhook",
            username="TestBot",
            timeout_sec=1.0,
            min_interval_sec=0.0,
        )
        fake_response = Mock()
        fake_response.status_code = 204
        fake_response.text = ""
        notifier._session.post = Mock(return_value=fake_response)  # type: ignore[attr-defined]

        ok = notifier.send(
            NotifyEvent(
                title="Test",
                description="hello",
                level="info",
            )
        )

        self.assertTrue(ok)
        notifier._session.post.assert_called_once()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
