import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import List
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from llama_index.core.schema import Document
from app.services.ingestion_service import IngestionService


@dataclass
class UrlIngestionResult:
    nodes_processed: int
    processed_urls: List[str]
    errors: List[str]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


class UrlIngestionService:
    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service

    def run_from_urls(self, urls: List[str]) -> UrlIngestionResult:
        clean_urls = [url.strip() for url in urls if url.strip()]
        if not clean_urls:
            raise ValueError("Provide at least one URL.")

        documents: List[Document] = []
        processed_urls: List[str] = []
        errors: List[str] = []

        for url in clean_urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError("Only http/https URLs are supported.")

                html = self._fetch_html(url)
                title = self._extract_title(html)
                text = self._extract_text(html)
                if not text.strip():
                    raise ValueError("No readable text found on the page.")

                metadata = {"source_url": url}
                if title:
                    metadata["title"] = title

                documents.append(Document(text=text, metadata=metadata))
                processed_urls.append(url)
            except Exception as exc:
                errors.append(f"{url} -> {exc}")

        if not documents:
            raise ValueError("No valid URLs could be processed.")

        nodes_processed = self.ingestion_service.index_documents(documents)
        return UrlIngestionResult(
            nodes_processed=nodes_processed,
            processed_urls=processed_urls,
            errors=errors,
        )

    def _fetch_html(self, url: str) -> str:
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (RAG URL Ingest)"},
        )
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                raise ValueError("URL did not return HTML or plain text content.")
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="ignore")

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        title = unescape(match.group(1))
        return re.sub(r"\s+", " ", title).strip()

    def _extract_text(self, html: str) -> str:
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        extractor.close()
        return extractor.get_text()
