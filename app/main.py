"""
FastAPI app: chat endpoint with retrieval + conversation.
Serves static chat page and streams LLM responses.
"""

import logging
import sys

# Ensure debug logs are visible in uvicorn console
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.conversation import generate, generate_stream
from src.response_renderer import process_response

app = FastAPI(title="Talk to the People")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BOOKS_DIR = PROJECT_ROOT / "books"
retriever = Retriever(books_dir=BOOKS_DIR)
TOP_K = 8
EMPTY_RESPONSE = "No relevant passages found."


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] | None = None


class ChatResponse(BaseModel):
    content: str


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat: retrieve, build template, generate, merge, render."""
    chunks = retriever.search(request.message, top_k=TOP_K)
    if not chunks:
        return ChatResponse(content=EMPTY_RESPONSE)
    raw = generate(
        request.message,
        chunks,
        history=request.history,
    )
    content = process_response(raw, chunks=chunks)
    return ChatResponse(content=content)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat: stage status + raw tokens + final formatted result."""

    def event_stream():
        yield "data: [status] Retrieving passages...\n\n"
        chunks = retriever.search(request.message, top_k=TOP_K)

        if not chunks:
            yield f"data: [final] {EMPTY_RESPONSE}\n\n"
            yield "data: [done]\n\n"
            return

        yield f"data: [status] Found {len(chunks)} passages. Analyzing...\n\n"

        raw_parts: list[str] = []
        for token in generate_stream(
            request.message,
            chunks,
            history=request.history,
        ):
            raw_parts.append(token)
            yield f"data: [token] {token}\n\n"

        raw = "".join(raw_parts)
        content = process_response(raw, chunks=chunks)

        yield "data: [final]\n"
        for line in content.split("\n"):
            yield f"data: {line}\n"
        yield "data: \n\n"
        yield "data: [done]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
