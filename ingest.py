"""
Ingest: chunk FAQ files → embed with VoyageAI → store in MongoDB.
Run once (or whenever FAQs change) before querying.
"""

from dotenv import load_dotenv
load_dotenv()

from pymongo.operations import SearchIndexModel
from rag_core import collection, load_and_chunk_faqs, embed_texts

# Load and chunk all FAQ markdown files
chunks = load_and_chunk_faqs()
print(f"Chunked {len(chunks)} pieces from FAQ files")

# Embed all chunks in one batch
embeddings = embed_texts([c["text"] for c in chunks])

# Clear existing data and insert fresh
collection.delete_many({})
collection.insert_many([
    {"text": c["text"], "source": c["source"], "embedding": e}
    for c, e in zip(chunks, embeddings)
])
print(f"Inserted {len(chunks)} documents into MongoDB")

# Create vector search index (skip if exists)
existing = [i["name"] for i in collection.list_search_indexes()]
if "vector_index" not in existing:
    collection.create_search_index(SearchIndexModel(
        definition={"fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 512,  # voyage-3-lite output
            "similarity": "cosine",
        }]},
        name="vector_index",
        type="vectorSearch",
    ))
    print("Created vector search index")
else:
    print("Vector search index already exists")
