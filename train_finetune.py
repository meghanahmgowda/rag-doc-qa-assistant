"""Optional: Fine-tune the Sentence-Transformer retriever on domain Q&A pairs.

Use this script if your project rubric requires explicit model training/fine-tuning.
After training, pass `--model models/finetuned-retriever` to ingest.py or set
EMBEDDING_MODEL_PATH=models/finetuned-retriever before starting app.py.
"""
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("models/finetuned-retriever")

# Replace or expand with your own domain (question, relevant_passage) pairs
TRAIN_PAIRS = [
    (
        "Why use RAG instead of fine-tuning an LLM?",
        "Documents can be updated or replaced without retraining the entire language model, "
        "and the generator is grounded in retrieved evidence to reduce hallucinations.",
    ),
    (
        "What does the text splitter do in a RAG pipeline?",
        "Text Splitter breaks long documents into overlapping chunks such as 400-600 characters "
        "with 80-100 character overlap so context is preserved.",
    ),
    (
        "Which vector stores are commonly used for RAG?",
        "Vector Store indexes dense vectors for fast similarity search using tools like FAISS, "
        "Chroma, or Pinecone.",
    ),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(BASE_MODEL)

    examples = [InputExample(texts=[q, p]) for q, p in TRAIN_PAIRS]
    dataloader = DataLoader(examples, shuffle=True, batch_size=2)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=3,
        warmup_steps=2,
        show_progress_bar=True,
    )
    model.save(str(OUTPUT_DIR))
    print(f"Saved fine-tuned retriever to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
