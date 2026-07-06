from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from search_r1_bm25.agent.reward import normalize_answer
from search_r1_bm25.retrieval.client import search


def contains_answer(text: str, gold_answers: list[str]) -> bool:
    norm_text = normalize_answer(text)
    return any(normalize_answer(answer) in norm_text for answer in gold_answers)


def evaluate_retriever(dataset_path: str, endpoint: str, top_k: int = 3, limit: int | None = None) -> dict:
    total = 0
    recalled = 0
    empty = 0
    latencies: list[float] = []
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and total >= limit:
                break
            item = json.loads(line)
            question = item["question"]
            gold_answers = item["answers"]
            t0 = time.perf_counter()
            payload = search(endpoint, question, top_k)
            latencies.append(time.perf_counter() - t0)
            docs = payload.get("results", [])
            if not docs:
                empty += 1
            joined = " ".join(f"{d.get('title', '')} {d.get('text', '')}" for d in docs)
            recalled += int(contains_answer(joined, gold_answers))
            total += 1
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    return {
        "total": total,
        "answer_recall_at_3": recalled / total if total else 0.0,
        "empty_rate": empty / total if total else 0.0,
        "mean_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p95_latency_ms": p95 * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8008/search")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_retriever(args.dataset, args.endpoint, args.top_k, args.limit), indent=2))


if __name__ == "__main__":
    main()
