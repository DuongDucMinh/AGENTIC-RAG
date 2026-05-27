"""Gradio interface mounted by the FastAPI app."""

from __future__ import annotations

import html
import json
import logging
from typing import Any

import gradio as gr

from app.services.chat_service import get_chat_service

logger = logging.getLogger(__name__)


def _format_citations(citations: list[dict[str, Any]]) -> list[list[str]]:
    """Convert citation dictionaries into Gradio table rows."""
    rows: list[list[str]] = []
    for citation in citations:
        rows.append(
            [
                str(citation.get("title") or ""),
                str(citation.get("so_ky_hieu") or ""),
                str(citation.get("article_number") or ""),
                str(citation.get("article_title") or ""),
                str(citation.get("status") or ""),
            ]
        )
    return rows


def _format_answer_html(answer: str, mode: str) -> str:
    """Render the final answer inside a highlighted HTML card."""
    tone_class = {
        "grounded_answer": "answer-card grounded",
        "clarification": "answer-card clarification",
        "out_of_domain": "answer-card out-of-domain",
        "insufficient_context": "answer-card insufficient",
        "error": "answer-card error",
    }.get(mode, "answer-card")
    safe_answer = html.escape(answer or "").replace("\n", "<br>")
    return (
        f'<section class="{tone_class}">'
        '<div class="answer-eyebrow">Answer</div>'
        f'<div class="answer-body">{safe_answer}</div>'
        "</section>"
    )


def _chat(question: str, session_id: str, debug: bool) -> tuple[str, list[list[str]], str, str]:
    """Run one chat turn and return answer, citations, trace, and mode."""
    clean_question = (question or "").strip()
    clean_session_id = (session_id or "gradio-demo").strip() or "gradio-demo"
    if not clean_question:
        mode = "clarification"
        return _format_answer_html("Vui lòng nhập câu hỏi.", mode), [], "{}", mode

    try:
        result = get_chat_service().chat(session_id=clean_session_id, question=clean_question, debug=debug)
    except Exception as exc:
        logger.exception("Gradio chat failed")
        mode = "error"
        return _format_answer_html(f"Lỗi khi xử lý câu hỏi: {exc}", mode), [], "{}", mode

    trace = result.get("retrieval_trace") or {}
    mode = str(trace.get("response_mode") or ("out_of_domain" if result.get("out_of_domain") else "unknown"))
    trace_json = json.dumps(trace, ensure_ascii=False, indent=2) if debug else "{}"
    return (
        _format_answer_html(str(result.get("answer") or ""), mode),
        _format_citations(result.get("citations") or []),
        trace_json,
        mode,
    )


def build_gradio_app() -> gr.Blocks:
    """Build the Gradio Blocks UI for the chat agent."""
    with gr.Blocks(
        title="Vietnamese Tax Legal RAG",
        analytics_enabled=False,
        css="""
        :root {
            --app-ink: #182028;
            --app-muted: #5d6a78;
            --app-paper: #f5f1e8;
            --app-accent: #0b6e4f;
            --app-accent-soft: #d8efe5;
            --app-border: #d3c7b7;
            --app-warning: #a44a1f;
            --app-warning-soft: #fde6d8;
            --app-danger: #8f1d2c;
            --app-danger-soft: #f9dde2;
            --app-info: #234e70;
            --app-info-soft: #ddebf5;
        }

        .gradio-container {
            background:
                radial-gradient(circle at top left, #f9f4ea 0%, rgba(249, 244, 234, 0) 35%),
                linear-gradient(180deg, #f0ebe2 0%, #e7dfd2 100%);
        }

        .answer-card {
            border: 1px solid var(--app-border);
            border-left: 8px solid var(--app-accent);
            border-radius: 18px;
            background: rgba(245, 241, 232, 0.96);
            box-shadow: 0 18px 40px rgba(24, 32, 40, 0.08);
            padding: 20px 22px;
            color: var(--app-ink);
        }

        .answer-card.clarification {
            border-left-color: var(--app-info);
            background: var(--app-info-soft);
        }

        .answer-card.out-of-domain,
        .answer-card.insufficient {
            border-left-color: var(--app-warning);
            background: var(--app-warning-soft);
        }

        .answer-card.error {
            border-left-color: var(--app-danger);
            background: var(--app-danger-soft);
        }

        .answer-eyebrow {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--app-muted);
            margin-bottom: 10px;
            font-weight: 700;
        }

        .answer-body {
            font-size: 17px;
            line-height: 1.75;
            font-weight: 500;
        }
        """,
    ) as demo:
        gr.Markdown(
            "# Vietnamese Tax Legal RAG\n"
            "Hỏi đáp về thuế, phí, lệ phí và nghĩa vụ tài chính dựa trên dữ liệu pháp luật đã index."
        )
        with gr.Row():
            session_id = gr.Textbox(label="Session ID", value="gradio-demo")
            debug = gr.Checkbox(label="Hiển thị retrieval trace", value=False)
        question = gr.Textbox(
            label="Câu hỏi",
            lines=3,
            placeholder="Ví dụ: tổ chức thu phí, lệ phí có trách nhiệm gì?",
        )
        submit = gr.Button("Gửi câu hỏi", variant="primary")
        mode = gr.Textbox(label="Response mode", interactive=False)
        answer = gr.HTML(label="Câu trả lời")
        citations = gr.Dataframe(
            headers=["Văn bản", "Số ký hiệu", "Điều", "Tiêu đề điều/mục", "Trạng thái"],
            datatype=["str", "str", "str", "str", "str"],
            label="Citations",
            interactive=False,
        )
        trace = gr.Code(label="Debug trace", language="json")

        submit.click(
            fn=_chat,
            inputs=[question, session_id, debug],
            outputs=[answer, citations, trace, mode],
        )
        question.submit(
            fn=_chat,
            inputs=[question, session_id, debug],
            outputs=[answer, citations, trace, mode],
        )
    return demo
