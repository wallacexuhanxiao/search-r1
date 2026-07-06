from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WIKI_MARKUP_RE = re.compile(r"(\{\{.*?\}\}|\[\[|\]\]|<[^>]+>)", re.S)
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = WIKI_MARKUP_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def chunk_words(text: str, min_words: int = 80, max_words: int = 180) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                part = words[i : i + max_words]
                if len(part) >= min_words:
                    chunks.append(" ".join(part))
            continue
        if len(current) + len(words) > max_words and len(current) >= min_words:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)
    if len(current) >= min_words:
        chunks.append(" ".join(current))
    return chunks


def convert_jsonl(input_path: str, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(input_path).open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            title = clean_text(str(obj.get("title", "")))
            text = clean_text(str(obj.get("text") or obj.get("contents") or ""))
            if not title or not text:
                continue
            for passage in chunk_words(text):
                doc_id = f"wiki_{n}"
                payload = {
                    "id": doc_id,
                    "title": title,
                    "text": passage,
                    "contents": f"{title}\n{passage}",
                }
                dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wiki jsonl articles to Pyserini passages.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
