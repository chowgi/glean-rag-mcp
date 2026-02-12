"""
RAG core: chunking, embedding (VoyageAI), vector store (MongoDB), generation (OpenAI).

Design choices:
- VoyageAI voyage-3-lite for embeddings: purpose-built for retrieval, outperforms
  OpenAI ada-002 on search benchmarks at lower dimensionality (512 dims).
- MongoDB Atlas vector search: production-grade persistence and scalability vs
  in-memory numpy (which loses data on restart and doesn't scale).
- gpt-4o-mini for generation: cost-efficient, fast, plenty capable for FAQ Q&A.
- ~200-char chunks: small enough for precise retrieval, large enough for context.
"""

import os
import json
from typing import Dict, List
from pathlib import Path

import voyageai
from pymongo import MongoClient
from openai import OpenAI

# --- Config (all from env vars) ---
FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-lite")  # 512 dims
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
MONGODB_DB = os.getenv("MONGODB_DB", "glean_rag")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "faq_chunks")

# --- Fail fast on missing keys ---
_VOYAGE_KEY = os.getenv("VOYAGE_API_KEY")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not _VOYAGE_KEY:
    raise RuntimeError("VOYAGE_API_KEY is not set")
if not _OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# --- Clients ---
voyage_client = voyageai.Client(api_key=_VOYAGE_KEY)
openai_client = OpenAI(api_key=_OPENAI_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]


# ---------------- Core utilities ----------------

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Split text into fixed-size character chunks. Simple and predictable."""
    text = text.strip()
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def load_and_chunk_faqs(faq_dir: str = FAQ_DIR) -> List[Dict]:
    """Load all .md files, chunk each, return list of {text, source} dicts."""
    chunks = []
    faq_path = Path(faq_dir)
    for md_file in sorted(faq_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        for chunk in chunk_text(content, CHUNK_SIZE):
            chunks.append({"text": chunk, "source": md_file.name})
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using VoyageAI. Returns list of 512-dim vectors."""
    result = voyage_client.embed(texts, model=EMBED_MODEL, input_type="document")
    return result.embeddings


def embed_query(q: str) -> List[float]:
    """Embed a single query using VoyageAI."""
    result = voyage_client.embed([q], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


def generate_answer(context: str, question: str) -> str:
    """Generate an answer grounded in the retrieved context, citing source files."""
    system_prompt = (
        "You are a helpful FAQ assistant. Answer the question using ONLY the provided context. "
        "Always cite at least 2 source filenames in your answer. "
        "If the context doesn't contain enough information, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low temp for factual answers
        max_tokens=512,
    )
    return response.choices[0].message.content


# ---------------- Public API ----------------

def ask_faq_core(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, object]:
    """Main entry point: retrieve relevant chunks and generate a cited answer."""
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")
    top_k = max(1, min(top_k, 10))

    # Embed the question
    q_embedding = embed_query(q)

    # Vector search in MongoDB
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": q_embedding,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "text": 1,
                "source": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = list(collection.aggregate(pipeline))

    if not results:
        return {"answer": "No relevant FAQ content found.", "sources": []}

    # Build context from retrieved chunks
    context_parts = [f"From {r['source']}:\n{r['text']}" for r in results]
    context = "\n\n".join(context_parts)

    # Generate answer
    answer = generate_answer(context, q)

    # Collect distinct source filenames (at least 2 if available)
    sources = sorted(set(r["source"] for r in results))

    return {"answer": answer, "sources": sources}


# ---------------- CLI runner ----------------
if __name__ == "__main__":
    q = input("Enter your question: ")
    print(json.dumps(ask_faq_core(q), indent=2))
