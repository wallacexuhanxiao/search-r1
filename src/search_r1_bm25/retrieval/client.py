from __future__ import annotations

import argparse
import json

import requests


def search(endpoint: str, query: str, top_k: int = 3) -> dict:
    response = requests.post(endpoint, json={"query": query, "top_k": top_k}, timeout=10)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8008/search")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(search(args.endpoint, args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
