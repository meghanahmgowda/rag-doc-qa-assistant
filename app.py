import json
import os
import shutil
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from ingest import build_index, DEFAULT_MODEL

UPLOAD_DIR = Path("uploaded_docs")
INDEX_DIR = Path("vector_store")
STATIC_DIR = Path("static")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")

app = FastAPI(title="Document Q&A Assistant (RAG)", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

embedder = None
generator = None
faiss_index = None
metadata: List[dict] = []


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)


def load_models():
    global embedder, generator
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if generator is None:
        generator = pipeline("text2text-generation", model=LLM_MODEL_NAME)


def load_vector_store():
    global faiss_index, metadata
    index_path = INDEX_DIR / "docs.index"
    meta_path = INDEX_DIR / "metadata.json"
    if index_path.exists() and meta_path.exists():
        faiss_index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup_event():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not (INDEX_DIR / "docs.index").exists() and Path("sample_data").exists():
        build_index(Path("sample_data"), INDEX_DIR, model_name=EMBEDDING_MODEL_NAME)
    load_models()
    load_vector_store()


@app.get("/")
def serve_home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stats")
def get_stats():
    sources = sorted({m["source"] for m in metadata})
    return {
        "indexed_chunks": len(metadata),
        "documents": sources,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
    }


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        dest = UPLOAD_DIR / Path(file.filename).name
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved.append(dest.name)

    stats = build_index(UPLOAD_DIR, INDEX_DIR, model_name=EMBEDDING_MODEL_NAME)
    load_vector_store()
    return {"uploaded": saved, **stats}


@app.post("/ask")
def ask_question(payload: AskRequest):
    if faiss_index is None or not metadata:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a file first.")

    load_models()
    query_vec = embedder.encode(
        [payload.question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(payload.top_k, len(metadata))
    scores, indices = faiss_index.search(query_vec, k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        retrieved.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "score": round(float(score), 4),
            "text": item["text"],
        })

    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
    context = context[:1400]

    prompt = (
        f"Question: {payload.question}\n\n"
        f"Context:\n{context}\n\n"
        "Write a detailed, complete explanation using the context above:"
    )

    result = generator(
        prompt,
        min_new_tokens=50,
        max_new_tokens=220,
        num_beams=4,
        length_penalty=1.5,
        repetition_penalty=1.2,
        do_sample=False,
        truncation=True,
    )
    answer = result[0]["generated_text"].strip()

    return {"question": payload.question, "answer": answer, "sources": retrieved}