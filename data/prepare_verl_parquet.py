from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are a search agent. Solve the question by reasoning, optionally calling "
    "the search tool, and then give the final answer. Use at most two searches."
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def convert_row(row: dict, split: str, index: int) -> dict:
    question = str(row["question"]).strip()
    answers = row.get("answers") or row.get("gold_answers") or row.get("golden_answers") or []
    if isinstance(answers, str):
        answers = [answers]
    answers = [str(answer).strip() for answer in answers if str(answer).strip()]
    data_source = row.get("dataset", "qa")
    ground_truth = {"target": answers}
    reward_model = {"style": "rule", "ground_truth": ground_truth}
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "search",
        "reward_model": reward_model,
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": tools_kwargs,
        },
        "metadata": {
            "dataset": data_source,
            "answers": answers,
        },
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
        },
    }


def convert_split(input_path: Path, output_path: Path, split: str) -> None:
    rows = [convert_row(row, split, idx) for idx, row in enumerate(read_jsonl(input_path))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(json.dumps({"split": split, "rows": len(rows), "output": str(output_path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="/root/autodl-tmp/search-r1-bm25/data/splits")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/search-r1-bm25/data/verl")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    mapping = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    for split, filename in mapping.items():
        convert_split(splits_dir / filename, output_dir / f"{split}.parquet", split)


if __name__ == "__main__":
    main()
