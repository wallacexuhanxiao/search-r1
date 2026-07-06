from search_r1_bm25.agent.parser import parse_model_output
from search_r1_bm25.agent.reward import final_answer_reward, normalize_answer
from search_r1_bm25.agent.search_tool import BM25SearchTool


def test_parse_search():
    parsed = parse_model_output("<think>x</think><search> Hemingway author </search>", search_turns_used=0)
    assert parsed.kind == "search"
    assert parsed.query == "Hemingway author"


def test_parse_answer_reward():
    text = "<think>x</think><answer>The Ernest Hemingway!</answer>"
    assert final_answer_reward(text, ["Ernest Hemingway"]) == 1
    assert normalize_answer("The Old, Man") == "old man"


def test_search_query_cleaning():
    tool = BM25SearchTool(max_query_tokens=3)
    assert tool.clean_query("<search> a b c d </search>") == "a b c"
