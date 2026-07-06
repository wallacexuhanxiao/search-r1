from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class RolloutTurn:
    thought: str
    query: str
    retrieved_docs: list[dict[str, Any]]


@dataclass
class RolloutRecord:
    question: str
    gold_answers: list[str]
    turns: list[RolloutTurn] = field(default_factory=list)
    final_answer: str = ""
    reward: int = 0
    error: str = ""


class RolloutLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RolloutRecord) -> None:
        payload = asdict(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
