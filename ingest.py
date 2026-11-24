import argparse
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("vector_store")


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += max(1, chunk_size - overlap)
    return chunks


def build_index(
    data_dir: Path,
    index_dir: Path = INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 500,
    overlap: int = 100,
) -> Dict[str, int]:
    index_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, str]] = []

    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            raw_text = extract_text_from_file(path)
            for idx, chunk in enumerate(chunk_text(raw_text, chunk_size, overlap)):
                records.append({
                    "source": path.name,
                    "chunk_id": idx,
                    "text": chunk,
                })

    if not records:
        raise RuntimeError(f"No supported .txt, .md, or .pdf files found in {data_dir}")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [r["text"] for r in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "docs.index"))
    (index_dir / "metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (index_dir / "config.json").write_text(
        json.dumps({"model_name": model_name, "chunk_size": chunk_size, "overlap": overlap}, indent=2),
        encoding="utf-8",
    )

    return {"files_indexed": len({r["source"] for r in records}), "chunks": len(records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index")
    parser.add_argument("--data_dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    stats = build_index(args.data_dir, model_name=args.model)
    print(f"Indexed {stats['files_indexed']} file(s) into {stats['chunks']} chunks.")
