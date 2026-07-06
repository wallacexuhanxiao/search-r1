from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import requests


SEARCH_TAG_RE = re.compile(r"</?search>", re.I)


@dataclass
class SearchResult:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class BM25SearchTool:
    endpoint: str = "http://127.0.0.1:8008/search"
    top_k: int = 3
    timeout: float = 10.0
    max_query_tokens: int = 64
    max_observation_tokens: int = 384
    cache: dict[str, list[SearchResult]] = field(default_factory=dict)

    def clean_query(self, query: str) -> str:
        query = SEARCH_TAG_RE.sub("", query).strip()
        tokens = query.split()
        return " ".join(tokens[: self.max_query_tokens]).strip()

    def search(self, query: str) -> list[SearchResult]:
        query = self.clean_query(query)
        if not query:
            raise ValueError("empty_query")
        if query in self.cache:
            return self.cache[query]

        response = requests.post(
            self.endpoint,
            json={"query": query, "top_k": self.top_k},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        results = [
            SearchResult(
                doc_id=str(item.get("doc_id", "")),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                score=float(item.get("score", 0.0)),
            )
            for item in payload.get("results", [])
        ]
        self.cache[query] = results
        return results

    def format_information(self, results: list[SearchResult]) -> str:
        budget = self.max_observation_tokens
        lines: list[str] = ["<information>"]
        used = 0
        for idx, result in enumerate(results, start=1):
            snippet_tokens = f"{result.title}. {result.text}".split()
            remaining = max(0, budget - used)
            if remaining == 0:
                break
            snippet = " ".join(snippet_tokens[:remaining])
            used += len(snippet.split())
            lines.append(f"[{idx}] {snippet}")
        lines.append("</information>")
        return "\n".join(lines)

    def __call__(self, query: str) -> tuple[str, list[SearchResult]]:
        results = self.search(query)
        return self.format_information(results), results
