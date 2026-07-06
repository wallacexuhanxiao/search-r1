from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def analyze(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return {"total": 0}
    rewards = [int(row.get("reward", 0)) for row in rows]
    search_turns = [len(row.get("turns", [])) for row in rows]
    repeated = 0
    for row in rows:
        queries = [turn.get("query", "") for turn in row.get("turns", [])]
        repeated += int(len(queries) != len(set(queries)))
    answer_rate = sum(bool(row.get("final_answer", "")) for row in rows) / len(rows)
    format_valid = sum(not row.get("error") for row in rows) / len(rows)
    return {
        "total": len(rows),
        "mean_reward": statistics.mean(rewards),
        "answer_rate": answer_rate,
        "search_rate": sum(t > 0 for t in search_turns) / len(rows),
        "average_search_turns": statistics.mean(search_turns),
        "repeated_query_rate": repeated / len(rows),
        "format_valid_rate": format_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trajectories), indent=2))


if __name__ == "__main__":
    main()
