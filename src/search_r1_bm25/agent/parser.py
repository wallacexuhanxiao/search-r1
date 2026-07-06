from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


TAG_RE = re.compile(r"<(?P<tag>think|search|answer)>\s*(?P<body>.*?)\s*</(?P=tag)>", re.S | re.I)


@dataclass(frozen=True)
class ParsedAction:
    kind: Literal["search", "answer", "invalid"]
    thought: str = ""
    query: str = ""
    answer: str = ""
    error: str = ""


def _all_tag_bodies(text: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    return [m.group(1).strip() for m in pattern.finditer(text)]


def parse_model_output(
    text: str,
    *,
    search_turns_used: int,
    max_search_turns: int = 2,
) -> ParsedAction:
    """Parse one model continuation into either a search action or a final answer."""
    thought = "\n\n".join(_all_tag_bodies(text, "think")).strip()
    searches = _all_tag_bodies(text, "search")
    answers = _all_tag_bodies(text, "answer")

    if "<search" in text.lower() and not searches:
        return ParsedAction(kind="invalid", thought=thought, error="unclosed_search_tag")
    if "<answer" in text.lower() and not answers:
        return ParsedAction(kind="invalid", thought=thought, error="unclosed_answer_tag")
    if searches and answers:
        return ParsedAction(kind="invalid", thought=thought, error="both_search_and_answer")
    if len(answers) > 1:
        return ParsedAction(kind="invalid", thought=thought, error="multiple_answers")
    if len(searches) > 1:
        return ParsedAction(kind="invalid", thought=thought, error="multiple_searches")

    if answers:
        answer = answers[-1].strip()
        if not answer:
            return ParsedAction(kind="invalid", thought=thought, error="empty_answer")
        return ParsedAction(kind="answer", thought=thought, answer=answer)

    if searches:
        if search_turns_used >= max_search_turns:
            return ParsedAction(kind="invalid", thought=thought, error="max_search_turns_exceeded")
        query = searches[-1].strip()
        if not query:
            return ParsedAction(kind="invalid", thought=thought, error="empty_query")
        return ParsedAction(kind="search", thought=thought, query=query)

    return ParsedAction(kind="invalid", thought=thought, error="no_action_tag")


def extract_final_answer(text: str) -> str | None:
    answers = _all_tag_bodies(text, "answer")
    if not answers:
        return None
    return answers[-1].strip()
