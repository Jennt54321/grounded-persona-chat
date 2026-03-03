# Evaluation

Evaluation pipeline for the RAG system: retrieval, generation, citation parsing, and metrics.

## Quick Start

**Use the project venv** so `accelerate` is available (otherwise generation will fail). Either activate it (`source .venv/bin/activate`) or call the venv Python explicitly:

```bash
# From project root; use venv Python (e.g. .venv/bin/python or activate venv first)
python -m eval.run_eval -q eval/questions_life.json
# Or explicitly:
.venv/bin/python -m eval.run_eval -q eval/questions_life.json

# Quick test with 3 questions
python -m eval.run_eval -q eval/questions_life.json -n 3

# Re-run metrics only from existing results (no retrieval, no generation)
python -m eval.run_eval --from-results eval/results/questions_life_results_checkpoint.json
python -m eval.run_eval -f eval/results/questions_life_results_checkpoint.json -n 5  # first 5 only
```

## LLM-as-a-Judge (Relevancy & Faithfulness)

在 run_eval 時可選跑 judge，對每筆結果打 **relevancy**（與問題相關度）與 **faithfulness**（與引用原文一致度），1–5 分：

```bash
python -m eval.run_eval -q eval/questions_life.json --run-judge
```

## Output

Output files use the questions file stem as prefix (e.g. `questions_life.json` → `questions_life_*`):

- `{stem}_results.json` - Per-question results with validity and diversity metrics
- `{stem}_summary.json` - Aggregate metrics
- `{stem}_full.json` - Full evaluation with metadata
- `{stem}_report.md` - Human-readable report
- `{stem}_results_checkpoint.json` - Checkpoint for `--from-results`
