"""RAG: chunk → embed (VoyageAI) → store/query (MongoDB) → answer (OpenAI)."""

import os, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # load API keys and MONGODB_URI from .env

import voyageai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from openai import OpenAI

# --- Config (env vars so you can override without editing code) ---
FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))  # folder of .md FAQ files
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))  # chars per chunk for embedding
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-lite")  # VoyageAI model, 512-dim output
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")

# Lazy-initialised clients (so MCP can start even if keys aren’t set yet)
_voyage = _openai = _coll = None

def _init():
    """Create VoyageAI, OpenAI, and MongoDB clients on first use."""
    global _voyage, _openai, _coll
    if _voyage is None:
        _voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        _openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _coll = MongoClient(MONGODB_URI)["glean_rag"]["faq_chunks"]

def get_collection():
    """Return the MongoDB collection (db: glean_rag, collection: faq_chunks), initialising clients if needed."""
    _init()
    return _coll

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
    _init()
    return _voyage.embed(texts, model=EMBED_MODEL, input_type="document").embeddings

def embed_query(q):
    """Embed a single query string (input_type=query for search). Returns one 512-dim vector."""
    _init()
    return _voyage.embed([q], model=EMBED_MODEL, input_type="query").embeddings[0]

def generate_answer(context, question):
    """Call OpenAI to answer the question using only the given context; low temp for factual replies."""
    _init()
    r = _openai.chat.completions.create(
        model=LLM_MODEL, temperature=0.2,
        messages=[
            {"role": "system", "content": "FAQ assistant. Answer ONLY from context. Be direct. Cite at least two source filenames in parentheses when available."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return r.choices[0].message.content

def ensure_vector_index():
    """Create MongoDB vector search index on field 'embedding' if it doesn’t exist (512 dims, cosine)."""
    coll = get_collection()
    names = [i["name"] for i in coll.list_search_indexes()]
    if "vector_index" not in names:
        coll.create_search_index(SearchIndexModel(
            definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": 512, "similarity": "cosine"}]},
            name="vector_index", type="vectorSearch",
        ))

def ask_faq_core(question, top_k=4):
    """Embed question → vector search in MongoDB → build context from top_k chunks → generate answer with OpenAI."""
    coll = get_collection()
    q_embed = embed_query(question)
    # MongoDB $vectorSearch: cosine similarity, return top_k docs
    results = list(coll.aggregate([
        {"$vectorSearch": {"index": "vector_index", "path": "embedding", "queryVector": q_embed, "numCandidates": top_k * 10, "limit": top_k}},
        {"$project": {"text": 1, "source": 1}},
    ]))
    context = "\n\n".join(f"From {r['source']}:\n{r['text']}" for r in results)
    answer = generate_answer(context, question)
    return {"answer": answer, "sources": sorted(set(r["source"] for r in results))}

if __name__ == "__main__":
    print(json.dumps(ask_faq_core(input("Question: ")), indent=2))
