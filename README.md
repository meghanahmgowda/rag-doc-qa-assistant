# Document Q&A Assistant with RAG (College Mini Project)

A full-stack **Retrieval-Augmented Generation (RAG)** web application that ingests PDF/TXT documents, builds a dense vector index (plus optional domain fine-tuning of the retriever), integrates with an LLM backend, and serves an interactive web UI with source citations.

## Project Architecture

1. **Data Collection & Preprocessing (`ingest.py`)**:
   - Extracts text from `.pdf` and `.txt` files.
   - Splits text into overlapping chunks (`chunk_size=500`, `overlap=100`) so context is preserved across boundaries.
2. **Model Training / Indexing (`ingest.py` & `train_finetune.py`)**:
   - **Vector Indexing**: Encodes chunks into 384-dimensional embeddings using `all-MiniLM-L6-v2` and stores them in a **FAISS** index.
   - **Optional Retriever Fine-Tuning (`train_finetune.py`)**: Fine-tunes the Sentence-Transformer embedding model on custom `(query, positive_passage)` pairs using `MultipleNegativesRankingLoss` to boost domain accuracy.
3. **Backend Integration (`app.py`)**:
   - FastAPI server exposing `/upload`, `/ask`, and `/stats`.
   - Retrieves top-k most similar chunks via cosine similarity, constructs a grounded prompt, and generates an answer using `google/flan-t5-base` (runs 100% locally, no API key needed).
4. **Frontend Website (`static/index.html`)**:
   - Clean, responsive HTML/CSS/JS UI for uploading documents, asking questions, adjusting `top_k`, and inspecting cited source chunks with similarity scores.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Index the included sample document via CLI
python ingest.py --data_dir sample_data

# 4. (Optional) Fine-tune the embedding model on your own Q&A pairs
python train_finetune.py

# 5. Start the web server
uvicorn app:app --reload --port 8000
```

Open **http://127.0.0.1:8000** in your browser.
