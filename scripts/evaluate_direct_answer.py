#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import string
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(tok for tok in text.split() if tok not in ARTICLES)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(str(answer)) for answer in answers)


def parse_obj(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def extract_question_and_answers(row) -> tuple[str, list[str], str]:
    data_source = str(row.get("data_source", "unknown"))
    extra = parse_obj(row.get("extra_info", {}))
    if isinstance(extra, dict) and extra.get("question"):
        question = str(extra["question"])
    else:
        prompt = parse_obj(row["prompt"])
        question = str(prompt[-1]["content"]) if isinstance(prompt, list) else str(prompt)

    answers: list[str] = []
    metadata = parse_obj(row.get("metadata", {}))
    if isinstance(metadata, dict) and "answers" in metadata:
        raw = metadata["answers"]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        answers = [str(x) for x in list(raw)]

    if not answers:
        reward_model = parse_obj(row.get("reward_model", {}))
        if isinstance(reward_model, dict):
            target = reward_model.get("ground_truth", {}).get("target", [])
            if hasattr(target, "tolist"):
                target = target.tolist()
            answers = [str(x) for x in list(target)]

    return question, answers, data_source


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    records = []
    prompts = []
    metas = []
    for _, row in df.iterrows():
        question, answers, data_source = extract_question_and_answers(row)
        messages = [
            {"role": "system", "content": "You are a helpful question answering assistant."},
            {
                "role": "user",
                "content": (
                    "Answer the question directly. Do not search. "
                    "Put only the final answer inside <answer></answer>.\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        metas.append((question, answers, data_source))

    for start in range(0, len(prompts), args.batch_size):
        outputs = llm.generate(prompts[start : start + args.batch_size], sampling)
        for offset, output in enumerate(outputs):
            question, answers, data_source = metas[start + offset]
            raw = output.outputs[0].text
            final = extract_final_answer(raw)
            score = 1 if exact_match(final, answers) else 0
            records.append(
                {
                    "dataset": data_source,
                    "question": question,
                    "gold_answers": answers,
                    "raw_response": raw,
                    "final_answer": final,
                    "score": score,
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {}
    for dataset in ["all"] + sorted({r["dataset"] for r in records}):
        subset = records if dataset == "all" else [r for r in records if r["dataset"] == dataset]
        summary[dataset] = {
            "total": len(subset),
            "em": sum(r["score"] for r in subset) / len(subset) if subset else 0.0,
            "answer_rate": sum(1 for r in subset if r["final_answer"]) / len(subset) if subset else 0.0,
        }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
