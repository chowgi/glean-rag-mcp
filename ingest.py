"""Ingestion pipeline: read FAQ .md files → chunk → embed → store in MongoDB. Run once (or when FAQs change)."""

import os
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from pymongo.operations import SearchIndexModel
load_dotenv(Path(__file__).parent / ".env")

from rag_core import get_collection

# Ingest-only constants (same embedding model as rag_core so vectors are comparable)
FAQ_DIR = Path(__file__).parent / "faqs"
CHUNK_SIZE = 200
EMBED_MODEL = "voyage-3-lite"

def chunk_text(text, size=CHUNK_SIZE):
    """Split text into fixed-size character chunks for embedding."""
    text = text.strip()
    return [text[i:i+size] for i in range(0, len(text), size)]

def load_and_chunk_faqs(faq_dir=FAQ_DIR):
    """Read all .md files in faq_dir, chunk each, return list of {text, source} dicts."""
    return [{"text": c, "source": md.name}
            for md in sorted(Path(faq_dir).glob("*.md"))
            for c in chunk_text(md.read_text())]

def embed_texts(texts):
    """Batch-embed strings with VoyageAI (input_type=document for long content). Returns list of 512-dim vectors."""
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return client.embed(texts, model=EMBED_MODEL, input_type="document").embeddings

def ensure_vector_index():
    """Create MongoDB vector search index on 'embedding' if it doesn’t exist (512 dims, cosine)."""
    coll = get_collection()
    names = [i["name"] for i in coll.list_search_indexes()]
    if "vector_index" not in names:
        coll.create_search_index(SearchIndexModel(
            definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": 512, "similarity": "cosine"}]},
            name="vector_index", type="vectorSearch",
        ))

# --- Pipeline: load → chunk → embed → store → ensure index ---
chunks = load_and_chunk_faqs()
embeddings = embed_texts([c["text"] for c in chunks])

coll = get_collection()
# Replace existing FAQ data with this run’s chunks
coll.delete_many({})
coll.insert_many([{"text": c["text"], "source": c["source"], "embedding": e} for c, e in zip(chunks, embeddings)])

# Create the vector search index if it doesn’t exist (needed for ask_faq queries)
ensure_vector_index()
print(f"Inserted {len(chunks)} documents")
