"""Chat endpoint backed by the LangGraph RAG agent."""

import logging

from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
# Goi LangGraph agent de tra loi cau hoi kem citations.
def chat(request: ChatRequest):
    """Answer a user question with grounded citations when context is available."""
    logger.info("Chat API requested session_id=%s", request.session_id)
    return get_chat_service().chat(session_id=request.session_id, question=request.question, debug=request.debug)
