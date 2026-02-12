"""
RAG core: chunking, embedding (VoyageAI), vector store (MongoDB), generation (OpenAI).

Design choices:
- VoyageAI voyage-3-lite for embeddings: purpose-built for retrieval, outperforms
  OpenAI ada-002 on search benchmarks at lower dimensionality (512 dims).
- MongoDB vector search: production-grade persistence and scalability vs
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

# ──────────────────────────────────────────────────────────────
# Configuration — everything reads from env vars with defaults
# ──────────────────────────────────────────────────────────────

FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))

# ~200 chars per chunk balances precision (small = focused) vs context (big = more info)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))

# How many chunks to retrieve by default. 4 gives enough context without noise.
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))

# VoyageAI voyage-3-lite: 512 dimensions, optimized for retrieval tasks.
# Cheaper and better at search than OpenAI's text-embedding-ada-002.
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-lite")

# gpt-4o-mini: best cost/quality tradeoff for FAQ answering.
# No need for gpt-4o when answers are short and grounded in context.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# MongoDB connection — defaults to localhost for Mac, override for Docker.
# directConnection=true is needed when MongoDB runs in a replica set
# with internal hostnames (common in Docker setups).
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
MONGODB_DB = os.getenv("MONGODB_DB", "glean_rag")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "faq_chunks")

# ──────────────────────────────────────────────────────────────
# Fail fast — crash immediately if keys are missing, not at query time
# ──────────────────────────────────────────────────────────────

_VOYAGE_KEY = os.getenv("VOYAGE_API_KEY")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not _VOYAGE_KEY:
    raise RuntimeError("VOYAGE_API_KEY is not set")
if not _OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# ──────────────────────────────────────────────────────────────
# Clients — initialized once at import time
# ──────────────────────────────────────────────────────────────

voyage_client = voyageai.Client(api_key=_VOYAGE_KEY)
openai_client = OpenAI(api_key=_OPENAI_KEY)

# Connect to MongoDB and get our collection handle
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]


# ──────────────────────────────────────────────────────────────
# STEP 1: Chunking — split FAQ text into ~200-char pieces
# ──────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """
    Simple fixed-size character chunking.
    
    Why not sentence splitting or overlap? Simplicity. The FAQ docs are short
    and well-structured, so fixed chunks work fine. In production you might
    use recursive text splitting or semantic chunking.
    """
    text = text.strip()
    if not text:
        return []
    # Slice the text into chunks of `size` characters each
    return [text[i:i + size] for i in range(0, len(text), size)]


def load_and_chunk_faqs(faq_dir: str = FAQ_DIR) -> List[Dict]:
    """
    Load every .md file in the FAQ directory, chunk it, and track which
    file each chunk came from (for citations later).
    """
    chunks = []
    faq_path = Path(faq_dir)

    # sorted() for deterministic ordering across runs
    for md_file in sorted(faq_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        for chunk in chunk_text(content, CHUNK_SIZE):
            # Each chunk remembers its source filename for citation
            chunks.append({"text": chunk, "source": md_file.name})

    return chunks


# ──────────────────────────────────────────────────────────────
# STEP 2: Embedding — convert text to vectors with VoyageAI
# ──────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of document texts using VoyageAI.
    
    input_type="document" tells Voyage these are passages to be searched,
    not queries. Voyage optimizes the embedding differently for each type.
    Returns list of 512-dimensional float vectors.
    """
    result = voyage_client.embed(texts, model=EMBED_MODEL, input_type="document")
    return result.embeddings


def embed_query(q: str) -> List[float]:
    """
    Embed a single query using VoyageAI.
    
    input_type="query" optimizes the embedding for retrieval (asymmetric search).
    This is a key VoyageAI feature — query and document embeddings are
    generated differently for better search performance.
    """
    result = voyage_client.embed([q], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


# ──────────────────────────────────────────────────────────────
# STEP 3: Generation — answer the question using retrieved context
# ──────────────────────────────────────────────────────────────

def generate_answer(context: str, question: str) -> str:
    """
    Send the retrieved chunks + question to gpt-4o-mini and get a grounded answer.
    
    The system prompt constrains the model to:
    1. Only use provided context (no hallucination)
    2. Always cite source filenames (traceability)
    3. Admit when it doesn't know
    """
    system_prompt = (
        "You are a helpful FAQ assistant. Answer the question using ONLY the "
        "provided context. Always cite at least 2 source filenames in your answer. "
        "If the context doesn't contain enough information, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low temp = more factual, less creative
        max_tokens=512,
    )
    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# Public API — the main RAG pipeline: embed → retrieve → generate
# ──────────────────────────────────────────────────────────────

def ask_faq_core(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, object]:
    """
    Main entry point. Takes a natural language question, returns an answer
    with source citations.
    
    Pipeline:
    1. Embed the question with VoyageAI
    2. Find the top-k most similar chunks in MongoDB via $vectorSearch
    3. Pass those chunks as context to gpt-4o-mini
    4. Return the answer + list of source filenames
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")

    # Clamp top_k to valid range
    top_k = max(1, min(top_k, 10))

    # --- Embed the question ---
    # This produces a 512-dim vector optimized for query-document matching
    q_embedding = embed_query(q)

    # --- Retrieve relevant chunks from MongoDB ---
    # $vectorSearch does approximate nearest neighbor (ANN) search.
    # numCandidates controls the breadth of the initial search (more = more accurate, slower).
    # We use 10x top_k as a reasonable trade-off.
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",       # Name of our vector search index
                "path": "embedding",            # Field containing the vectors
                "queryVector": q_embedding,     # Our question vector
                "numCandidates": top_k * 10,   # ANN search breadth
                "limit": top_k,                 # Final number of results
            }
        },
        {
            # Project only the fields we need, plus the similarity score
            "$project": {
                "text": 1,
                "source": 1,
                "score": {"$meta": "vectorSearchScore"},  # Cosine similarity
            }
        },
    ]
    results = list(collection.aggregate(pipeline))

    if not results:
        return {"answer": "No relevant FAQ content found.", "sources": []}

    # --- Build context string for the LLM ---
    # Each chunk is labeled with its source file so the LLM can cite them
    context_parts = [f"From {r['source']}:\n{r['text']}" for r in results]
    context = "\n\n".join(context_parts)

    # --- Generate the answer ---
    answer = generate_answer(context, q)

    # --- Collect unique source filenames ---
    sources = sorted(set(r["source"] for r in results))

    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────────────────────
# CLI runner — quick way to test without the MCP server
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = input("Enter your question: ")
    print(json.dumps(ask_faq_core(q), indent=2))
