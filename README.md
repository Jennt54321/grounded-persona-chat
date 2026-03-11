# Talk to the People — A Grounded Persona Conversation System

Chat with an assistant grounded in Plato's dialogues (Apology, Meno, Gorgias, Republic). Uses semantic retrieval over chunk embeddings and Qwen2.5-3B-Instruct via Hugging Face for grounded responses (local inference, no external API). Retrieval uses BGE bi-encoder (BAAI/bge-base-en-v1.5) plus BGE reranker (BAAI/bge-reranker-base): bi-encoder returns top 50 candidates, cross-encoder reranks to top 5.

## Project structure

- **app/** — FastAPI app: `app/main.py`, `app/static/index.html`
- **src/** — Core logic: retriever (BGE bi-encoder + BGE reranker cross-encoder), conversation (Qwen2.5), `citation_utils.py`, `response_renderer.py`
- **scripts/** — Chunking and embedding: `chunk_apology.py`, `chunk_meno.py`, `chunk_gorgias.py`, `chunk_republic.py`, `embed_chunks.py`, `verify_*_chunks.py`
- **books/** — Source `.txt` and chunk JSONs: `apology.txt`, `meno.txt`, `gorgias.txt`, `republic.txt` and (after running scripts) `*_chunks.json`
- **eval/** — Evaluation pipeline: `run_eval.py`, `questions_life.json`, supporting modules (`citation_parser.py`, `chunk_index.py`, `metrics.py`, `llm_judge.py`, `compute_retrieval_similarity.py`, `run_contextual_metrics.py`), results
- **docs/** — Research design and evaluation records

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

Retrieval traces are saved to `retrieval_logs/` during chat sessions.

## API

- **GET /** — Serves the chat UI (static HTML)
- **POST /chat** — Non-streaming: retrieval → generation → response as bullet list with citations. Body: `{ "message": "...", "history": [...] }` (optional)
- **POST /chat/stream** — SSE stream: status messages, then final formatted result with the same request body

## Evaluation

Evaluation pipeline for the RAG system: retrieval, generation, citation parsing, and metrics.

**Use the project venv** so `accelerate` is available (otherwise generation will fail). Either activate it (`source .venv/bin/activate`) or call the venv Python explicitly:

```bash
# From project root
python -m eval.run_eval -q eval/questions_life.json

# Or explicitly:
.venv/bin/python -m eval.run_eval -q eval/questions_life.json

# Quick test with 3 questions
python -m eval.run_eval -q eval/questions_life.json -n 3

# Re-run metrics only from existing results (no retrieval, no generation)
python -m eval.run_eval --from-results eval/results/questions_life_results_checkpoint.json
python -m eval.run_eval -f eval/results/questions_life_results_checkpoint.json -n 5  # first 5 only

# Skip LLM-as-a-judge (runs by default)
python -m eval.run_eval -q eval/questions_life.json --no-run-judge
```

### LLM-as-a-Judge (Relevancy & Faithfulness)

By default, the judge scores each result for **relevancy** (question relevance) and **faithfulness** (alignment with cited passages), 1–5. Use `--no-run-judge` to skip.

```bash
python -m eval.run_eval -q eval/questions_life.json --run-judge
```

### Output

Output files are written to `eval/results/` (or `--output`) and use the questions file stem as prefix (e.g. `questions_life.json` → `questions_life_*`):

- `{stem}_retrieval.json` — Cached retrieval results (Stage 1)
- `{stem}_results_checkpoint.json` — Checkpoint for `--from-results`
- `{stem}_results.json` — Per-question results with validity and diversity metrics
- `{stem}_summary.json` — Aggregate metrics
- `{stem}_full.json` — Full evaluation with metadata
- `{stem}_report.md` — Human-readable report

## Research

See [docs/Research Design and Evaluation Records.md](docs/Research%20Design%20and%20Evaluation%20Records.md) for research hypotheses, metrics, and evaluation records.
