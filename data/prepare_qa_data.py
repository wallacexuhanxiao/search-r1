from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


WS_RE = re.compile(r"\s+")


def norm_question(question: str) -> str:
    return WS_RE.sub(" ", question).strip().lower()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_nq_open() -> list[dict]:
    ds = load_dataset("google-research-datasets/nq_open")
    rows = []
    for split in ds:
        for item in ds[split]:
            answers = item.get("answer") or item.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            question = item["question"].strip()
            if question and answers:
                rows.append({"dataset": "nq", "question": question, "answers": list(answers)})
    return rows


def load_hotpotqa() -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    rows = []
    for split in ds:
        for item in ds[split]:
            question = item["question"].strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                rows.append({"dataset": "hotpotqa", "question": question, "answers": [answer]})
    return rows


def dedupe(rows: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for row in rows:
        key = norm_question(row["question"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def split_rows(rows: list[dict], train_n: int, val_n: int, test_n: int, rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = rows[:]
    rng.shuffle(shuffled)
    needed = train_n + val_n + test_n
    if len(shuffled) < needed:
        raise ValueError(f"not enough rows: have {len(shuffled)}, need {needed}")
    return shuffled[:train_n], shuffled[train_n : train_n + val_n], shuffled[train_n + val_n : needed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/search-r1-bm25/data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nq-train", type=int, default=2000)
    parser.add_argument("--hotpot-train", type=int, default=2000)
    parser.add_argument("--nq-val", type=int, default=500)
    parser.add_argument("--hotpot-val", type=int, default=500)
    parser.add_argument("--nq-test", type=int, default=500)
    parser.add_argument("--hotpot-test", type=int, default=500)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    nq = dedupe(load_nq_open(), seen)
    hotpot = dedupe(load_hotpotqa(), seen)

    nq_train, nq_val, nq_test = split_rows(nq, args.nq_train, args.nq_val, args.nq_test, rng)
    hp_train, hp_val, hp_test = split_rows(hotpot, args.hotpot_train, args.hotpot_val, args.hotpot_test, rng)

    output = Path(args.output_dir)
    write_jsonl(output / "train.jsonl", nq_train + hp_train)
    write_jsonl(output / "validation.jsonl", nq_val + hp_val)
    write_jsonl(output / "test.jsonl", nq_test + hp_test)
    write_jsonl(output / "nq_validation.jsonl", nq_val)
    write_jsonl(output / "hotpotqa_validation.jsonl", hp_val)
    write_jsonl(output / "nq_test.jsonl", nq_test)
    write_jsonl(output / "hotpotqa_test.jsonl", hp_test)

    print(json.dumps({
        "nq": len(nq),
        "hotpotqa": len(hotpot),
        "train": len(nq_train) + len(hp_train),
        "validation": len(nq_val) + len(hp_val),
        "test": len(nq_test) + len(hp_test),
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
