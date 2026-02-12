"""
RAG core: chunking, embedding (VoyageAI), vector store (MongoDB), generation (OpenAI).

- VoyageAI voyage-3-lite for embeddings: purpose-built for retrieval, 512 dims
- MongoDB $vectorSearch: persistent vector store with cosine similarity
- gpt-4o-mini for generation: cost-efficient for FAQ Q&A
"""

import os, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import voyageai
from pymongo import MongoClient
from openai import OpenAI

# --- Config from env vars ---
FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-lite")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")

# --- Clients ---
voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
collection = MongoClient(MONGODB_URI)["glean_rag"]["faq_chunks"]


def chunk_text(text, size=CHUNK_SIZE):
    """Split text into fixed-size character chunks."""
    text = text.strip()
    return [text[i:i+size] for i in range(0, len(text), size)]


def load_and_chunk_faqs(faq_dir=FAQ_DIR):
    """Load all .md files and chunk them. Returns list of {text, source} dicts."""
    chunks = []
    for md in sorted(Path(faq_dir).glob("*.md")):
        for chunk in chunk_text(md.read_text()):
            chunks.append({"text": chunk, "source": md.name})
    return chunks


def embed_texts(texts):
    """Batch embed documents using VoyageAI. Returns list of 512-dim vectors."""
    return voyage.embed(texts, model=EMBED_MODEL, input_type="document").embeddings


def embed_query(q):
    """Embed a single query. VoyageAI uses input_type to optimise for retrieval."""
    return voyage.embed([q], model=EMBED_MODEL, input_type="query").embeddings[0]


def generate_answer(context, question):
    """Call OpenAI to answer using only the retrieved context, citing sources."""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,  # low temp for factual answers
        messages=[
            {"role": "system", "content":
                "You are an FAQ assistant. Answer using ONLY the provided context. "
                "Be direct — state the most relevant facts. Don't ask follow-up questions. "
                "Infer intent: e.g. 'locked out' relates to password reset. "
                "Cite source filenames in parentheses."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


def ask_faq_core(question, top_k=4):
    """Main entry: embed question → vector search MongoDB → generate cited answer."""
    q_embedding = embed_query(question)

    # MongoDB vector search — cosine similarity, return top_k chunks
    results = list(collection.aggregate([
        {"$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": q_embedding,
            "numCandidates": top_k * 10,
            "limit": top_k,
        }},
        {"$project": {"text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))

    # Build context string from retrieved chunks
    context = "\n\n".join(f"From {r['source']}:\n{r['text']}" for r in results)
    answer = generate_answer(context, question)
    sources = sorted(set(r["source"] for r in results))

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    print(json.dumps(ask_faq_core(input("Question: ")), indent=2))
