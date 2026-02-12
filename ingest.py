"""
Ingest script: chunks FAQ files, embeds with VoyageAI, stores in MongoDB.
Run this once (or whenever FAQs change) before querying.

Usage:
    python ingest.py
"""

import os
import sys
from pathlib import Path

# Load .env if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_core import (
    collection, load_and_chunk_faqs, embed_texts, FAQ_DIR,
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION,
)


def create_vector_search_index():
    """Create the MongoDB vector search index if it doesn't exist."""
    from pymongo.operations import SearchIndexModel

    # Check if index already exists
    existing = list(collection.list_search_indexes())
    for idx in existing:
        if idx.get("name") == "vector_index":
            print("✓ Vector search index 'vector_index' already exists")
            return

    index_def = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 512,  # voyage-3-lite output dim
                    "similarity": "cosine",
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    collection.create_search_index(index_def)
    print("✓ Created vector search index 'vector_index'")


def ingest():
    """Main ingestion pipeline."""
    print(f"FAQ directory: {FAQ_DIR}")
    print(f"MongoDB: {MONGODB_URI} → {MONGODB_DB}.{MONGODB_COLLECTION}")

    # 1. Load and chunk
    chunks = load_and_chunk_faqs()
    print(f"✓ Loaded {len(chunks)} chunks from FAQ files")

    if not chunks:
        print("No chunks found. Check FAQ_DIR.")
        sys.exit(1)

    # 2. Embed all chunks (batch)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with VoyageAI...")
    embeddings = embed_texts(texts)
    print(f"✓ Got {len(embeddings)} embeddings ({len(embeddings[0])} dims)")

    # 3. Store in MongoDB (clear existing, then insert)
    collection.delete_many({})  # Fresh start
    docs = []
    for chunk, embedding in zip(chunks, embeddings):
        docs.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": embedding,
        })
    collection.insert_many(docs)
    print(f"✓ Inserted {len(docs)} documents into MongoDB")

    # 4. Create vector search index
    create_vector_search_index()

    print("\n✅ Ingestion complete! You can now query with mcp_server.py or rag_core.py")


if __name__ == "__main__":
    ingest()
