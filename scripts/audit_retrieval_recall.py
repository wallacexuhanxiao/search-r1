#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def contains_answer(text: str, gold_answers: list[str]) -> bool:
    norm_text = normalize_answer(text)
    return any(normalize_answer(answer) in norm_text for answer in gold_answers if answer)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def extract_question(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"]).strip()
    env_kwargs = row.get("env_kwargs")
    if isinstance(env_kwargs, dict) and env_kwargs.get("question"):
        return str(env_kwargs["question"]).strip()
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
    return str(row.get("question", "")).strip()


def extract_dataset(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("dataset"):
        return str(metadata["dataset"])
    return str(row.get("data_source") or row.get("dataset") or "unknown")


def extract_gold_answers(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        answers = as_list(metadata.get("answers"))
        if answers:
            return [str(answer).strip() for answer in answers if str(answer).strip()]

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth") or {}
        answers = as_list(ground_truth.get("target"))
        if answers:
            return [str(answer).strip() for answer in answers if str(answer).strip()]

    answers = row.get("answers") or row.get("gold_answers") or row.get("golden_answers")
    return [str(answer).strip() for answer in as_list(answers) if str(answer).strip()]


def load_validation_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = df.to_dict(orient="records")
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_trajectory_queries(path: Path) -> dict[str, dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            queries: list[str] = []
            for turn in item.get("turns", []):
                query = str(turn.get("query", "")).strip()
                if query:
                    queries.append(query)
            by_question[question] = {
                "queries": queries,
                "reward": item.get("reward"),
                "final_answer": item.get("final_answer"),
            }
    return by_question


def search(endpoint: str, query: str, top_k: int, timeout: float) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    response = requests.post(endpoint, json={"query": query, "top_k": top_k}, timeout=timeout)
    response.raise_for_status()
    elapsed = time.perf_counter() - t0
    return list(response.json().get("results", [])), elapsed


def docs_text(docs: list[dict[str, Any]], k: int) -> str:
    return " ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs[:k])


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def write_summary(rows: list[dict[str, Any]], latencies: list[float], output_summary: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
        if row.get("success") is True:
            groups.setdefault("success", []).append(row)
        elif row.get("success") is False:
            groups.setdefault("failure", []).append(row)

    summary = []
    for name, part in groups.items():
        summary.append(
            {
                "group": name,
                "total": len(part),
                "hit@1": mean_bool(part, "hit@1"),
                "hit@3": mean_bool(part, "hit@3"),
                "hit@5": mean_bool(part, "hit@5"),
                "empty_rate": sum(int(row["num_docs"] == 0) for row in part) / len(part) if part else 0.0,
            }
        )

    payload = {
        "summary": summary,
        "mean_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000 if latencies else 0.0,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether BM25 results for validation questions or model-generated queries contain gold answers."
    )
    parser.add_argument("--validation-parquet", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/search")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieve up to this K so hit@1/3/5 can be computed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--trajectories", help="Optional JSONL rollout log. If set, use model-generated turn queries.")
    parser.add_argument("--output-csv", default="retrieval_recall_audit.csv")
    parser.add_argument("--output-summary", default="retrieval_recall_audit_summary.json")
    args = parser.parse_args()

    validation_rows = load_validation_rows(Path(args.validation_parquet), args.limit)
    trajectory_map = load_trajectory_queries(Path(args.trajectories)) if args.trajectories else {}

    audit_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    max_k = max(args.top_k, 5)

    for row in validation_rows:
        question = extract_question(row)
        gold_answers = extract_gold_answers(row)
        dataset = extract_dataset(row)
        trajectory = trajectory_map.get(question, {})
        queries = trajectory.get("queries") or [question]
        success = None
        if "reward" in trajectory and trajectory["reward"] is not None:
            success = bool(float(trajectory["reward"]) > 0)

        for turn_idx, query in enumerate(queries, start=1):
            docs, latency = search(args.endpoint, query, max_k, args.timeout)
            latencies.append(latency)
            audit_rows.append(
                {
                    "dataset": dataset,
                    "question": question,
                    "gold_answer": " | ".join(gold_answers),
                    "query": query,
                    "turn": turn_idx,
                    "success": success,
                    "hit@1": contains_answer(docs_text(docs, 1), gold_answers),
                    "hit@3": contains_answer(docs_text(docs, 3), gold_answers),
                    "hit@5": contains_answer(docs_text(docs, 5), gold_answers),
                    "num_docs": len(docs),
                    "top1_doc_id": docs[0].get("doc_id", "") if docs else "",
                    "top1_title": docs[0].get("title", "") if docs else "",
                    "latency_ms": round(latency * 1000, 3),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()) if audit_rows else [])
        if audit_rows:
            writer.writeheader()
            writer.writerows(audit_rows)

    write_summary(audit_rows, latencies, Path(args.output_summary))
    print(json.dumps({"rows": len(audit_rows), "csv": str(output_csv), "summary": args.output_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
