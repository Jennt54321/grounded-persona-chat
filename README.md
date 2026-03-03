# Talk to the People — A Grounded Persona Conversation System

Chat with an assistant grounded in Plato's dialogues (Apology, Meno, Gorgias, Republic). Uses semantic retrieval over chunk embeddings and Qwen2.5-3B-Instruct via Hugging Face for grounded responses (local inference, no external API). Retrieval uses BGE bi-encoder (BAAI/bge-base-en-v1.5) plus BGE reranker (BAAI/bge-reranker-base): bi-encoder returns top 50 candidates, reranker returns top 5.

## Project structure

- **app/** — FastAPI app: `app/main.py`, `app/static/index.html`
- **src/** — Core logic: retriever (BGE + reranker), conversation (Qwen2.5), `citation_utils.py`, `response_renderer.py`
- **scripts/** — Chunking and embedding: `chunk_apology.py`, `chunk_meno.py`, `chunk_gorgias.py`, `chunk_republic.py`, `embed_chunks.py`, `generate_questions.py`, `verify_*_chunks.py`
- **books/** — Source `.txt` and chunk JSONs: `apology.txt`, `meno.txt`, `gorgias.txt`, `republic.txt` and (after running scripts) `*_chunks.json`
- **eval/** — Evaluation pipeline; see [eval/README.md](eval/README.md)

## Prerequisites

- Python 3.10+
- GPU recommended (for faster inference)

For GPU (e.g. Google Colab), see [colab.ipynb](colab.ipynb).

## Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare chunk files and embeddings. Chunk files (`books/*_chunks.json`) are produced from source texts and are not shipped in the repo. If you only have `books/*.txt`:

   - Run the chunk scripts first: `python scripts/chunk_apology.py`, `python scripts/chunk_meno.py`, `python scripts/chunk_gorgias.py`, `python scripts/chunk_republic.py`
   - Then add embeddings: `python scripts/embed_chunks.py`

   If you already have `*_chunks.json` with no embeddings, run:

   ```bash
   python scripts/embed_chunks.py
   ```

## Run

1. Start the web app:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Open http://localhost:8000 in your browser.

## API

- **GET /** — Serves the chat UI (static HTML)
- **POST /chat** — Non-streaming: retrieval → generation → response as bullet list with citations. Body: `{ "message": "...", "history": [...] }` (optional)
- **POST /chat/stream** — SSE stream: status messages, then final formatted result with the same request body

## Evaluation

See [eval/README.md](eval/README.md) for the evaluation pipeline. Quick start:

```bash
python -m eval.run_eval -q eval/questions_life.json
```

To generate new evaluation questions: `python scripts/generate_questions.py`
