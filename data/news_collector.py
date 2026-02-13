"""Automated market news collector for RAG context."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import html
import logging
import os
import re
from typing import Any
import xml.etree.ElementTree as ET

import requests

from core.config_loader import NewsConfig
from data.rag_store import NewsItem, SimpleRAGStore


LOGGER = logging.getLogger(__name__)


class MarketNewsCollector:
    """Collect notices/news and upsert into SimpleRAGStore."""

    def __init__(
        self,
        exchange: Any,
        rag_store: SimpleRAGStore,
        config: NewsConfig,
    ) -> None:
        self.exchange = exchange
        self.rag_store = rag_store
        self.config = config
        self._session = requests.Session()

    def collect_once(self, watchlist: list[str]) -> dict[str, Any]:
        symbols = sorted({self._asset_symbol(code) for code in watchlist})
        documents: list[NewsItem] = []
        source_counts: dict[str, int] = {}

        if self.config.use_bithumb_notice:
            notice_docs = self._fetch_bithumb_notices(symbols=symbols)
            documents.extend(notice_docs)
            source_counts["bithumb_notice"] = len(notice_docs)

        if self.config.use_coindesk_rss:
            coindesk_docs = self._fetch_coindesk_rss(symbols=symbols)
            documents.extend(coindesk_docs)
            source_counts["coindesk_rss"] = len(coindesk_docs)

        if self.config.use_naver_openapi:
            naver_docs = self._fetch_naver_openapi_news(symbols=symbols)
            documents.extend(naver_docs)
            source_counts["naver_openapi"] = len(naver_docs)

        if documents:
            self.rag_store.upsert(documents)

        return {
            "stored": len(documents),
            "source_counts": source_counts,
        }

    def _fetch_bithumb_notices(self, symbols: list[str]) -> list[NewsItem]:
        notices = self.exchange.get_bithumb_notices(
            page=1,
            limit=self.config.per_source_limit,
        )
        docs: list[NewsItem] = []
        for row in notices:
            title = str(
                row.get("title")
                or row.get("subject")
                or row.get("notice_title")
                or ""
            ).strip()
            if not title:
                continue
            body = str(row.get("content") or row.get("summary") or "").strip()
            url = str(row.get("url") or row.get("link") or "").strip()
            notice_id = str(row.get("id") or row.get("notice_id") or "")
            ts = self._parse_datetime(
                row.get("created_at")
                or row.get("created")
                or row.get("date")
                or row.get("published_at"),
            )
            for ticker in self._infer_tickers(title=title, summary=body, symbols=symbols):
                docs.append(
                    NewsItem(
                        id=f"bithumb_notice:{notice_id or hash((title, ts, ticker))}",
                        ticker=ticker,
                        title=title,
                        summary=body[:600],
                        source="bithumb_notice",
                        published_at=ts,
                        url=url,
                    )
                )
        return docs

    def _fetch_coindesk_rss(self, symbols: list[str]) -> list[NewsItem]:
        url = self.config.coindesk_rss_url
        xml_text = self._http_get_text(url)
        if not xml_text:
            return []

        docs: list[NewsItem] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            LOGGER.warning("Failed to parse CoinDesk RSS: %s", exc)
            return []

        items = root.findall(".//item")
        for item in items[: self.config.per_source_limit]:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            summary = self._strip_html(item.findtext("description") or "")
            link = (item.findtext("link") or "").strip()
            pub_date = self._parse_datetime(item.findtext("pubDate"))
            for ticker in self._infer_tickers(title=title, summary=summary, symbols=symbols):
                docs.append(
                    NewsItem(
                        id=f"coindesk:{hash((title, link, ticker))}",
                        ticker=ticker,
                        title=title,
                        summary=summary[:600],
                        source="coindesk_rss",
                        published_at=pub_date,
                        url=link,
                    )
                )
        return docs

    def _fetch_naver_openapi_news(self, symbols: list[str]) -> list[NewsItem]:
        client_id = os.getenv("NAVER_CLIENT_ID", "").strip()
        client_secret = os.getenv("NAVER_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return []

        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        docs: list[NewsItem] = []
        for symbol in symbols[:5]:
            query = self._symbol_to_ko_keyword(symbol)
            try:
                resp = self._session.get(
                    "https://openapi.naver.com/v1/search/news.json",
                    params={
                        "query": query,
                        "display": min(self.config.per_source_limit, 20),
                        "sort": "date",
                    },
                    headers=headers,
                    timeout=8,
                )
                data = resp.json() if resp.status_code < 500 else {}
            except Exception:
                continue
            items = data.get("items", []) if isinstance(data, dict) else []
            for row in items:
                title = self._strip_html(str(row.get("title", "")))
                if not title:
                    continue
                summary = self._strip_html(str(row.get("description", "")))
                link = str(row.get("originallink") or row.get("link") or "")
                ts = self._parse_datetime(row.get("pubDate"))
                docs.append(
                    NewsItem(
                        id=f"naver:{hash((title, link, symbol))}",
                        ticker=symbol,
                        title=title,
                        summary=summary[:600],
                        source="naver_openapi",
                        published_at=ts,
                        url=link,
                    )
                )
        return docs

    def _http_get_text(self, url: str) -> str:
        try:
            resp = self._session.get(url, timeout=8)
            if resp.status_code >= 400:
                return ""
            return resp.text
        except Exception:
            return ""

    @staticmethod
    def _asset_symbol(code: str) -> str:
        value = code.upper().strip()
        if "-" in value:
            return value.split("-")[-1]
        for quote in ("USDT", "USDC", "KRW", "BTC", "ETH"):
            if value.endswith(quote) and len(value) > len(quote):
                return value[: -len(quote)]
        return value

    def _infer_tickers(self, title: str, summary: str, symbols: list[str]) -> list[str]:
        text = f"{title} {summary}".upper()
        matched = [symbol for symbol in symbols if symbol and symbol in text]
        if matched:
            return matched[:3]
        if "BITCOIN" in text or "비트코인" in text:
            return ["BTC"]
        if "ETHEREUM" in text or "이더리움" in text:
            return ["ETH"]
        return ["MARKET"]

    @staticmethod
    def _parse_datetime(raw_value: Any) -> str:
        if raw_value is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw_value, datetime):
            dt = raw_value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()

        text = str(raw_value).strip()
        if not text:
            return datetime.now(timezone.utc).isoformat()
        try:
            return parsedate_to_datetime(text).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _strip_html(value: str) -> str:
        no_tag = re.sub(r"<[^>]+>", " ", value or "")
        return html.unescape(re.sub(r"\s+", " ", no_tag)).strip()

    @staticmethod
    def _symbol_to_ko_keyword(symbol: str) -> str:
        mapping = {
            "BTC": "비트코인",
            "ETH": "이더리움",
            "XRP": "리플",
            "SOL": "솔라나",
            "DOGE": "도지코인",
        }
        return mapping.get(symbol, symbol)
