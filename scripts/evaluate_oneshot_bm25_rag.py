#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import string
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return bool(pred) and any(pred == normalize_answer(str(answer)) for answer in answers)


def contains_answer(text: str, answers: list[str]) -> bool:
    norm_text = normalize_answer(text)
    return any(normalize_answer(str(answer)) in norm_text for answer in answers if str(answer).strip())


def parse_obj(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def extract_question_and_answers(row: dict[str, Any]) -> tuple[str, list[str], str]:
    data_source = str(row.get("data_source", "unknown"))
    extra = parse_obj(row.get("extra_info", {}))
    if isinstance(extra, dict) and extra.get("question"):
        question = str(extra["question"]).strip()
    else:
        prompt = parse_obj(row["prompt"])
        question = str(prompt[-1]["content"]).strip() if isinstance(prompt, list) else str(prompt).strip()

    answers: list[str] = []
    metadata = parse_obj(row.get("metadata", {}))
    if isinstance(metadata, dict) and "answers" in metadata:
        answers = [str(x).strip() for x in as_list(metadata["answers"]) if str(x).strip()]

    if not answers:
        reward_model = parse_obj(row.get("reward_model", {}))
        if isinstance(reward_model, dict):
            target = reward_model.get("ground_truth", {}).get("target", [])
            answers = [str(x).strip() for x in as_list(target) if str(x).strip()]

    return question, answers, data_source


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def search(endpoint: str, query: str, top_k: int, timeout: float) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    response = requests.post(endpoint, json={"query": query, "top_k": top_k}, timeout=timeout)
    response.raise_for_status()
    elapsed = time.perf_counter() - t0
    return list(response.json().get("results", [])), elapsed


def format_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "No BM25 results were returned."
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        blocks.append(f"[{idx}] {title}\n{text}")
    return "\n\n".join(blocks)


def summarize(records: list[dict[str, Any]], output: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    for record in records:
        groups.setdefault(str(record["dataset"]), []).append(record)

    payload = {}
    for name, subset in groups.items():
        latencies = [float(r["bm25_latency_ms"]) for r in subset]
        payload[name] = {
            "total": len(subset),
            "em": sum(int(r["score"]) for r in subset) / len(subset) if subset else 0.0,
            "answer_rate": sum(1 for r in subset if r["final_answer"]) / len(subset) if subset else 0.0,
            "retrieval_hit@1": sum(int(r["hit@1"]) for r in subset) / len(subset) if subset else 0.0,
            "retrieval_hit@3": sum(int(r["hit@3"]) for r in subset) / len(subset) if subset else 0.0,
            "empty_rate": sum(1 for r in subset if r["num_docs"] == 0) / len(subset) if subset else 0.0,
            "mean_bm25_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p95_bm25_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
            "avg_generation_tokens": statistics.mean([int(r["generation_tokens"]) for r in subset]) if subset else 0.0,
        }

    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-shot BM25 RAG with the original question as the query.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/search")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    rows = pd.read_parquet(args.data).to_dict(orient="records")
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    metas: list[dict[str, Any]] = []

    for row in rows:
        question, answers, data_source = extract_question_and_answers(row)
        docs, latency = search(args.endpoint, question, args.top_k, args.timeout)
        context = format_context(docs)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful question answering assistant. Answer only from the provided BM25 evidence.",
            },
            {
                "role": "user",
                "content": (
                    "Use the BM25 evidence to answer the question. "
                    "Put only the final answer inside <answer></answer>.\n\n"
                    f"Question: {question}\n\nBM25 evidence:\n{context}"
                ),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        doc_text = " ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs)
        metas.append(
            {
                "dataset": data_source,
                "question": question,
                "gold_answers": answers,
                "docs": docs,
                "hit@1": contains_answer(
                    " ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs[:1]), answers
                ),
                "hit@3": contains_answer(doc_text, answers),
                "num_docs": len(docs),
                "bm25_latency_ms": round(latency * 1000, 3),
            }
        )

    for start in range(0, len(prompts), args.batch_size):
        outputs = llm.generate(prompts[start : start + args.batch_size], sampling)
        for offset, output in enumerate(outputs):
            meta = metas[start + offset]
            raw = output.outputs[0].text
            final = extract_final_answer(raw)
            score = 1 if exact_match(final, meta["gold_answers"]) else 0
            records.append(
                {
                    "dataset": meta["dataset"],
                    "question": meta["question"],
                    "gold_answers": meta["gold_answers"],
                    "query": meta["question"],
                    "retrieved_docs": [
                        {
                            "doc_id": doc.get("doc_id", ""),
                            "title": doc.get("title", ""),
                            "text": doc.get("text", ""),
                            "bm25_score": doc.get("score", 0.0),
                        }
                        for doc in meta["docs"]
                    ],
                    "raw_response": raw,
                    "final_answer": final,
                    "score": score,
                    "hit@1": meta["hit@1"],
                    "hit@3": meta["hit@3"],
                    "num_docs": meta["num_docs"],
                    "bm25_latency_ms": meta["bm25_latency_ms"],
                    "generation_tokens": len(output.outputs[0].token_ids),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summarize(records, output.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
