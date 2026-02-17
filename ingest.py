"""Chunk FAQs → embed → store in MongoDB. Run once (or when FAQs change)."""

from rag_core import get_collection, load_and_chunk_faqs, embed_texts, ensure_vector_index

# Load all .md files from faqs/ and split into fixed-size chunks
chunks = load_and_chunk_faqs()
# Turn each chunk into a 512-dim vector via VoyageAI
embeddings = embed_texts([c["text"] for c in chunks])

coll = get_collection()
# Replace existing FAQ data with this run’s chunks
coll.delete_many({})
coll.insert_many([{"text": c["text"], "source": c["source"], "embedding": e} for c, e in zip(chunks, embeddings)])

# Create the vector search index if it doesn’t exist (needed for ask_faq queries)
ensure_vector_index()
print(f"Inserted {len(chunks)} documents")
