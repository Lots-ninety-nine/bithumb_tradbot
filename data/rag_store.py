"""Simple JSON-based RAG store for market news/context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(slots=True)
class NewsItem:
    id: str
    ticker: str
    title: str
    summary: str
    source: str
    published_at: str
    url: str = ""

    def score(self, query: str) -> int:
        q = query.lower().strip()
        haystack = f"{self.ticker} {self.title} {self.summary}".lower()
        return haystack.count(q) if q else 0


class SimpleRAGStore:
    """JSON-backed storage and retrieval.

    This keeps the interface stable until FAISS/vector DB is added.
    """

    def __init__(self, store_path: str = "data/rag_store.json") -> None:
        self.store_path = Path(store_path)
        self.items: list[NewsItem] = []
        self.load()

    def load(self) -> None:
        if not self.store_path.exists():
            self.items = []
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.items = [NewsItem(**item) for item in raw]
        except Exception:
            self.items = []

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in self.items]
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, documents: list[NewsItem]) -> None:
        by_id = {item.id: item for item in self.items}
        for doc in documents:
            by_id[doc.id] = doc
        self.items = sorted(
            by_id.values(),
            key=lambda x: x.published_at,
            reverse=True,
        )
        self.save()

    def latest_for_ticker(self, ticker: str, limit: int = 3) -> list[dict]:
        ticker = ticker.upper()
        selected = [item for item in self.items if item.ticker.upper() == ticker]
        if not selected:
            selected = [
                item
                for item in self.items
                if ticker.lower() in f"{item.title} {item.summary}".lower()
            ]
        return [asdict(item) for item in selected[:limit]]

    def query_for_trade(self, ticker: str, limit: int = 3) -> list[dict]:
        """Trade-time context query with ticker keyword fallback."""
        direct = self.latest_for_ticker(ticker=ticker, limit=limit)
        if direct:
            return direct
        return self.search(query=ticker, limit=limit)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        ranked = sorted(
            self.items,
            key=lambda x: (x.score(query), x.published_at),
            reverse=True,
        )
        return [asdict(item) for item in ranked[:limit]]

    @staticmethod
    def make_item(
        ticker: str,
        title: str,
        summary: str,
        source: str,
        url: str = "",
        published_at: str | None = None,
    ) -> NewsItem:
        ts = published_at or datetime.now(timezone.utc).isoformat()
        doc_id = f"{ticker}-{hash((title, ts, source))}"
        return NewsItem(
            id=doc_id,
            ticker=ticker.upper(),
            title=title,
            summary=summary,
            source=source,
            published_at=ts,
            url=url,
        )
