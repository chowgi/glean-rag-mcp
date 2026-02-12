"""
Ingest script: chunks FAQ files, embeds with VoyageAI, stores in MongoDB.

Run this once (or whenever FAQs change) before querying.
Kept separate from query-time code so the server starts fast.

Usage:
    python ingest.py
"""

import os
import sys
from pathlib import Path

# Load .env file if present — handy for local dev on Mac
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, env vars work fine

from rag_core import (
    collection, load_and_chunk_faqs, embed_texts, FAQ_DIR,
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION,
)


def create_vector_search_index():
    """
    Create the MongoDB vector search index for cosine similarity.
    
    This index enables $vectorSearch queries. It needs:
    - The field path ("embedding")
    - The number of dimensions (512 for voyage-3-lite)
    - The similarity metric (cosine — standard for text embeddings)
    
    Skips creation if the index already exists (safe to re-run).
    """
    from pymongo.operations import SearchIndexModel

    # Check if index already exists — don't duplicate
    existing = list(collection.list_search_indexes())
    for idx in existing:
        if idx.get("name") == "vector_index":
            print("✓ Vector search index 'vector_index' already exists")
            return

    # Define the vector search index
    index_def = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",         # Field where we store vectors
                    "numDimensions": 512,         # voyage-3-lite output size
                    "similarity": "cosine",       # Standard for text embeddings
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    collection.create_search_index(index_def)
    print("✓ Created vector search index 'vector_index'")


def ingest():
    """
    Main ingestion pipeline:
    1. Load and chunk FAQ markdown files
    2. Embed all chunks with VoyageAI (single batch call)
    3. Store chunks + embeddings in MongoDB
    4. Create the vector search index
    """
    print(f"FAQ directory: {FAQ_DIR}")
    print(f"MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")

    # --- Step 1: Load and chunk ---
    chunks = load_and_chunk_faqs()
    print(f"✓ Loaded {len(chunks)} chunks from FAQ files")

    if not chunks:
        print("No chunks found. Check FAQ_DIR.")
        sys.exit(1)

    # --- Step 2: Embed all chunks in one batch ---
    # VoyageAI handles batching internally, so we can send all at once
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with VoyageAI...")
    embeddings = embed_texts(texts)
    print(f"✓ Got {len(embeddings)} embeddings ({len(embeddings[0])} dims)")

    # --- Step 3: Store in MongoDB ---
    # Clear existing data for a clean slate (idempotent re-runs)
    collection.delete_many({})
    docs = []
    for chunk, embedding in zip(chunks, embeddings):
        docs.append({
            "text": chunk["text"],          # The chunk text (for context building)
            "source": chunk["source"],       # Source filename (for citations)
            "embedding": embedding,          # 512-dim vector (for search)
        })
    collection.insert_many(docs)
    print(f"✓ Inserted {len(docs)} documents into MongoDB")

    # --- Step 4: Create vector search index ---
    create_vector_search_index()

    print("\n✅ Ingestion complete! You can now query with mcp_server.py or rag_core.py")


if __name__ == "__main__":
    ingest()
