from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(min_length=1)
    topk: int = Field(default=3, ge=1, le=20)
    return_scores: bool = True


class SearchResponse(BaseModel):
    results: list[dict]


def create_app(index_dir: str) -> FastAPI:
    app = FastAPI(title="Search-R1 BM25 Retriever")

    @lru_cache(maxsize=1)
    def searcher():
        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyserini is required for BM25 search") from exc
        return LuceneSearcher(str(Path(index_dir)))

    def run_search(query: str, top_k: int) -> list[dict]:
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="empty_query")
        hits = searcher().search(query, k=top_k)
        results = []
        for hit in hits:
            raw = searcher().doc(hit.docid).raw()
            import json

            doc = json.loads(raw)
            contents = doc.get("contents", "")
            title, _, text = contents.partition("\n")
            results.append(
                {
                    "doc_id": doc.get("id", hit.docid),
                    "title": doc.get("title", title),
                    "text": doc.get("text", text or contents),
                    "score": hit.score,
                }
            )
        return results

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        return SearchResponse(results=run_search(request.query, request.top_k))

    @app.post("/retrieve")
    async def retrieve(request: Request) -> dict:
        payload = await request.json()
        if "queries" in payload:
            queries = payload.get("queries") or []
        elif "query" in payload:
            queries = [payload.get("query")]
        else:
            queries = []

        topk = payload.get("topk", payload.get("top_k", 3))
        return_scores = payload.get("return_scores", True)
        try:
            topk = max(1, min(int(topk), 20))
        except Exception:
            topk = 3

        batched_results = []
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                batched_results.append([])
                continue
            docs = run_search(query, topk)
            batched_results.append([
                {
                    "document": {
                        "id": doc["doc_id"],
                        "contents": doc["title"] + "\n" + doc["text"],
                    },
                    "score": doc["score"] if return_scores else None,
                }
                for doc in docs
            ])
        return {"result": batched_results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.index_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
