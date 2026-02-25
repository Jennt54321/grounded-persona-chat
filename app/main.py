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
from src.conversation import generate_values_batch_stream
from src.response_renderer import render_quotes_to_bullets

app = FastAPI(title="Talk to the People")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BOOKS_DIR = PROJECT_ROOT / "books"
RETRIEVAL_LOGS_DIR = PROJECT_ROOT / "retrieval_logs"
retriever = Retriever(books_dir=BOOKS_DIR)
TOP_K = 5
BI_TOP_K = 50
FINAL_TOP_K = 5  # Cross-encoder takes 5 from bi's 50
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
    """Non-streaming chat: Bi-encoder -> Cross-encoder rerank, batch value generation, render."""
    chunks, _ = retriever.search_with_rerank(
        request.message,
        bi_top_k=BI_TOP_K,
        final_top_k=FINAL_TOP_K,
        save_trace_dir=RETRIEVAL_LOGS_DIR,
    )
    if not chunks:
        return ChatResponse(content=EMPTY_RESPONSE)
    data, errors = None, []
    for event in generate_values_batch_stream(request.message, chunks):
        if event[0] == "done":
            payload = event[1]
            data = {"quotes": payload.get("quotes", [])}
            errors = payload.get("errors", [])
    content = render_quotes_to_bullets(data)
    if errors:
        err_summary = f"\n\n[注意] {len(errors)} 個 passage 分析時發生錯誤（可能為 context 長度超限）"
        content = content + err_summary
    return ChatResponse(content=content)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat: status (batch analysis), then final formatted result."""

    def event_stream():
        yield "data: [status] Retrieving passages...\n\n"
        chunks, _ = retriever.search_with_rerank(
            request.message,
            bi_top_k=BI_TOP_K,
            final_top_k=FINAL_TOP_K,
            save_trace_dir=RETRIEVAL_LOGS_DIR,
        )

        if not chunks:
            yield f"data: [final] {EMPTY_RESPONSE}\n\n"
            yield "data: [done]\n\n"
            return

        yield f"data: [status] Found {len(chunks)} passages. Analyzing...\n\n"

        data = None
        errors = []
        for event in generate_values_batch_stream(request.message, chunks):
            if event[0] == "status":
                _, i, total = event
                yield f"data: [status] Citation {i}/{total} done...\n\n"
            elif event[0] == "done":
                payload = event[1]
                data = {"quotes": payload.get("quotes", [])}
                errors = payload.get("errors", [])

        content = render_quotes_to_bullets(data)
        if errors:
            content += f"\n\n[注意] {len(errors)} 個 passage 分析時發生錯誤（可能為 context 長度超限）"
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
