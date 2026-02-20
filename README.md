# Talk to the People — A Grounded Persona Conversation System

Chat with an assistant grounded in Plato's dialogues (Apology, Meno, Gorgias, Republic). Uses semantic retrieval over chunk embeddings and Phi-3.5 via Ollama for grounded responses.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with Phi-3.5

## Setup

1. Install Ollama and pull the model:

   ```bash
   ollama pull phi3.5
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure chunk files have embeddings (run if not already done):

   ```bash
   python scripts/embed_chunks.py
   ```

## Run

1. Start Ollama (if not already running): `ollama run phi3.5` or keep the Ollama service running.
2. Start the web app:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Open http://localhost:8000 in your browser.
