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
    label = {
        "grounded_answer": "Grounded answer",
        "clarification": "Clarification",
        "out_of_domain": "Out of domain",
        "insufficient_context": "Insufficient context",
        "error": "Error",
    }.get(mode, "Response")
    return (
        f'<section class="{tone_class}">'
        f'<div class="answer-eyebrow">{label}</div>'
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
            --app-ink: #17212b;
            --app-muted: #5a6877;
            --app-bg: #eef2f0;
            --app-panel: #fbfcfa;
            --app-soft: #e2ebe6;
            --app-accent: #1f6f5b;
            --app-accent-2: #8c5f2f;
            --app-border: #cfd8d3;
            --app-warning: #95592d;
            --app-warning-soft: #fff1e5;
            --app-danger: #9f2f3f;
            --app-danger-soft: #ffe7eb;
            --app-info: #315f7c;
            --app-info-soft: #e7f1f6;
        }

        .gradio-container {
            background:
                linear-gradient(135deg, rgba(31, 111, 91, 0.12), rgba(140, 95, 47, 0.10)),
                linear-gradient(180deg, #f7f9f7 0%, var(--app-bg) 100%);
            color: var(--app-ink);
            font-family: ui-sans-serif, "Segoe UI", sans-serif;
        }

        .app-shell {
            max-width: 1180px;
            margin: 0 auto;
        }

        .app-hero {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: rgba(251, 252, 250, 0.92);
            padding: 22px 24px;
            margin-bottom: 16px;
            box-shadow: 0 14px 34px rgba(23, 33, 43, 0.08);
        }

        .app-title {
            font-size: 30px;
            line-height: 1.15;
            font-weight: 780;
            margin: 0 0 8px;
            color: var(--app-ink);
        }

        .app-subtitle {
            color: var(--app-muted);
            font-size: 15px;
            line-height: 1.55;
            margin: 0;
        }

        .app-panel {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: rgba(251, 252, 250, 0.94);
            padding: 16px;
        }

        .compact-mode textarea {
            font-size: 15px !important;
            line-height: 1.5 !important;
        }

        .answer-card {
            border: 1px solid var(--app-border);
            border-left: 6px solid var(--app-accent);
            border-radius: 8px;
            background: var(--app-panel);
            box-shadow: 0 10px 24px rgba(23, 33, 43, 0.07);
            padding: 18px 20px;
            color: var(--app-ink);
            min-height: 150px;
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
            letter-spacing: 0.08em;
            color: var(--app-muted);
            margin-bottom: 8px;
            font-weight: 700;
        }

        .answer-body {
            font-size: 16px;
            line-height: 1.7;
            font-weight: 500;
        }

        .primary-send button {
            min-height: 44px;
            border-radius: 6px !important;
            font-weight: 700 !important;
        }

        .gr-dataframe, .gr-code {
            border-radius: 8px !important;
        }
        """,
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="app-hero">
                    <h1 class="app-title">Vietnamese Tax Legal RAG</h1>
                    <p class="app-subtitle">
                        Grounded legal assistant for Vietnamese tax, fees, charges, and registration-fee questions.
                        Answers include source citations and optional retrieval traces for inspection.
                    </p>
                </section>
                """
            )
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["app-panel", "compact-mode"]):
                    question = gr.Textbox(
                        label="Question",
                        lines=5,
                        placeholder="Ví dụ: Nhà đất phải nộp lệ phí trước bạ theo mức phần trăm nào?",
                    )
                    with gr.Row():
                        session_id = gr.Textbox(label="Session ID", value="gradio-demo", scale=2)
                        debug = gr.Checkbox(label="Show retrieval trace", value=False, scale=1)
                    submit = gr.Button("Submit question", variant="primary", elem_classes=["primary-send"])
                    mode = gr.Textbox(label="Response mode", interactive=False)
                with gr.Column(scale=7):
                    answer = gr.HTML(label="Answer")
            citations = gr.Dataframe(
                headers=["Document", "Symbol", "Article", "Article/section title", "Status"],
                datatype=["str", "str", "str", "str", "str"],
                label="Citations",
                interactive=False,
                wrap=True,
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
