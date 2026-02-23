# Evaluation

Evaluation pipeline for the RAG system: retrieval, generation, citation parsing, and metrics (including RAGAS).

## Quick Start

```bash
# Pull the RAGAS judge model (Prometheus 2) before first run
ollama pull tensortemplar/prometheus2:7b

# Run evaluation (RAGAS enabled by default)
python -m eval.run_eval -q eval/questions_life.json

# Quick test with 3 questions
python -m eval.run_eval -q eval/questions_life.json -n 3

# Re-run metrics only from existing results (no retrieval, no generation)
python -m eval.run_eval --from-results eval/results/eval_results.json
python -m eval.run_eval -f eval/results/eval_results.json -n 5  # first 5 only
```

## RAGAS Metrics

RAGAS metrics use **Prometheus 2** as the LLM-as-judge via Ollama (local, free). Metrics computed:

| Metric | Description |
|--------|-------------|
| **Contextual Precision** | Retriever ranks relevant chunks higher (uses response or reference) |
| **Contextual Recall** | Retriever retrieves all necessary info (requires `reference` in questions) |
| **Contextual Relevancy** | Retrieved contexts are pertinent to the question |
| **Answer Relevancy** | Answer is on-topic and helpful |
| **Faithfulness** | Answer is grounded in retrieved context (no hallucination) |

### Configuration

- **Judge model**: Default `tensortemplar/prometheus2:7b` via Ollama
- **Ollama**: Ensure Ollama is running; set `OLLAMA_BASE_URL` if not `http://localhost:11434`
- **Override model**: `--ragas-llm <model>` (e.g. `phi3.5` for a different Ollama model)
- **Disable RAGAS**: `--no-ragas`
- **Chunk truncation**: `RAGAS_CHUNK_MAX_CHARS` (default 600) to stay within context limits

### Ground Truth for Context Recall

Add optional `reference` to questions in `questions.json` for Context Recall:

```json
{
  "id": 1,
  "question": "What does Socrates say about the charges?",
  "reference": "Socrates addresses the charges of corrupting the youth and impiety..."
}
```

Context Recall runs only on questions that have `reference`.

## Output

- `eval_results.json` - Per-question results including RAGAS scores
- `eval_summary.json` - Aggregate metrics
- `eval_full.json` - Full evaluation with metadata
- `eval_report.md` - Human-readable report

## Trust and Limitations

RAGAS metrics rely on an LLM judge. Scores can vary by model and run. Use them for relative comparison and trend tracking. Cross-check with the rule-based metrics (A1–A4) and spot-check samples manually when important.
