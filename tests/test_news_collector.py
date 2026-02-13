from __future__ import annotations

import unittest

from core.config_loader import NewsConfig
from data.news_collector import MarketNewsCollector
from data.rag_store import SimpleRAGStore


class _FakeExchange:
    def get_bithumb_notices(self, page: int = 1, limit: int = 20):
        return [
            {
                "id": "1",
                "title": "[공지] BTC 입출금 안내",
                "content": "비트코인 네트워크 점검",
                "created_at": "2026-02-13T00:00:00+00:00",
                "url": "https://example.com/notice/1",
            }
        ]


class NewsCollectorTest(unittest.TestCase):
    def test_collect_once_ingests_documents(self) -> None:
        exchange = _FakeExchange()
        rag_store = SimpleRAGStore(store_path="/tmp/test_rag_store_news.json")
        rag_store.items = []
        cfg = NewsConfig(
            enabled=True,
            use_bithumb_notice=True,
            use_coindesk_rss=False,
            use_naver_openapi=False,
            per_source_limit=5,
        )
        collector = MarketNewsCollector(exchange=exchange, rag_store=rag_store, config=cfg)
        result = collector.collect_once(watchlist=["KRW-BTC"])
        self.assertGreaterEqual(result["stored"], 1)
        self.assertGreaterEqual(len(rag_store.items), 1)


if __name__ == "__main__":
    unittest.main()
