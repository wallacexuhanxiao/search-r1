from __future__ import annotations

import re
import string
from collections.abc import Iterable

from .parser import extract_final_answer


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)
WS_RE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = ARTICLES_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def exact_match(prediction: str, gold_answers: Iterable[str]) -> bool:
    pred = normalize_answer(prediction)
    return bool(pred) and any(pred == normalize_answer(gold) for gold in gold_answers)


def final_answer_reward(model_text: str, gold_answers: Iterable[str]) -> int:
    answer = extract_final_answer(model_text)
    if answer is None or not answer.strip():
        return 0
    return int(exact_match(answer, gold_answers))
