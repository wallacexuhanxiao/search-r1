from __future__ import annotations

import gradio as gr

from search_r1_bm25.agent.search_tool import BM25SearchTool


def run_demo(question: str, model_choice: str) -> str:
    tool = BM25SearchTool()
    query = question.strip()
    try:
        information, results = tool(query)
    except Exception as exc:
        return f"Retriever error: {exc}"
    lines = [f"Model: {model_choice}", "", f"Question: {question}", "", "Turn 1:", f"Search Query: {query}", ""]
    for idx, result in enumerate(results, start=1):
        lines.append(f"[{idx}] {result.title} | score={result.score:.3f}")
        lines.append(result.text[:500])
        lines.append("")
    lines.append("Final Answer: load a base or GRPO model in evaluation/evaluate.py for full generation.")
    lines.append("")
    lines.append(information)
    return "\n".join(lines)


with gr.Blocks(title="Search-R1 BM25 Demo") as demo:
    gr.Markdown("# Search-R1 BM25 Demo")
    question = gr.Textbox(label="Question")
    model_choice = gr.Radio(["Base Model", "GRPO Model"], value="Base Model", label="Model")
    output = gr.Textbox(label="Trajectory", lines=24)
    gr.Button("Run").click(run_demo, inputs=[question, model_choice], outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
