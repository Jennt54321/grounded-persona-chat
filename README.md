# Talk to the People — A Grounded Persona Conversation System

Chat with an assistant grounded in Plato's dialogues (Apology, Meno, Gorgias, Republic). Uses semantic retrieval over chunk embeddings and Qwen2.5-3B-Instruct via Hugging Face for grounded responses (local inference, no external API).

## Prerequisites

- Python 3.10+
- GPU recommended (for faster inference)

## Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure chunk files have embeddings (run if not already done):

   ```bash
   python scripts/embed_chunks.py
   ```

## Run

1. Start the web app:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Open http://localhost:8000 in your browser.
