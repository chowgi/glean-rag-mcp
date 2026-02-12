# FAQ RAG + MCP Tool

A RAG prototype that answers natural-language questions from FAQ documents using vector search and LLM generation, exposed as an MCP tool.

## What We Built

A three-stage RAG pipeline:

1. **Ingest** — FAQ markdown files are chunked (~200 chars), embedded with VoyageAI, and stored in MongoDB
2. **Retrieve** — User questions are embedded and matched against stored chunks using MongoDB `$vectorSearch` (cosine similarity)
3. **Generate** — Top matching chunks are passed as context to an LLM, which generates a cited answer

The whole thing is wrapped as an MCP tool (`ask_faq`) so any MCP-compatible client (Claude Desktop, Cursor) can call it directly.

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | VoyageAI `voyage-3-lite` (512d) | Purpose-built for retrieval; outperforms OpenAI ada-002 on search benchmarks |
| **Vector Store** | MongoDB `$vectorSearch` | Persistent, scalable, production-realistic — vs in-memory numpy which loses data on restart |
| **LLM** | OpenAI `gpt-4o-mini` | Cost-efficient, fast, plenty capable for FAQ Q&A |
| **MCP** | stdio transport | Standard for local MCP tools |

## How It Works

```
Question → VoyageAI embed → MongoDB $vectorSearch → Top-K chunks → OpenAI generate → Cited answer
```

- **Chunking:** Fixed ~200 character splits. Simple and predictable for a small corpus.
- **Retrieval:** Cosine similarity via MongoDB vector search index (HNSW). Returns top 4 chunks by default.
- **Generation:** System prompt enforces grounded answers — no hallucination, must cite source filenames, infers intent (e.g. "locked out" → password reset).
- **Lazy client init:** API clients connect on first query, not at server startup — so the MCP server registers tools cleanly even before calling any APIs.

## How to Run

### 1. Install & configure
```bash
pip install -r requirements.txt
cp .env.example .env    # Add your API keys
```

### 2. Ingest the FAQ corpus
```bash
python ingest.py
```

### 3. Test via CLI
```bash
python rag_core.py
```

### 4. Run as MCP tool
Add to your MCP client config (Claude Desktop / Cursor):
```json
{
  "mcpServers": {
    "faq-rag": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {
        "VOYAGE_API_KEY": "your-key",
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

## Example Questions

These show the system understands **intent**, not just keywords:

| Question | What it tests |
|----------|---------------|
| "How do I reset my password?" | Direct keyword match (faq_auth.md) |
| "I'm locked out of my account" | Semantic inference — no "password" or "reset" in query |
| "Can I take 3 weeks off in a row?" | Retrieves the 2-week approval rule from PTO policy |
| "When do my shares kick in?" | Maps "shares" → equity vesting schedule |
| "I want to use one login for everything" | Maps to SSO without mentioning it |
| "What do new employees need to know?" | Cross-document retrieval from multiple FAQ files |

## Deviations from Starter Skeleton

The starter used OpenAI embeddings + in-memory numpy for cosine similarity. We replaced both:

- **VoyageAI instead of OpenAI embeddings** — Voyage models are purpose-built for retrieval and rank higher on search benchmarks (MTEB). Using a separate embedding provider also decouples retrieval quality from the LLM choice.
- **MongoDB instead of numpy** — A real vector database with persistence, indexing (HNSW), and `$vectorSearch` aggregation. Data survives restarts, and the same approach scales to millions of documents without code changes.
- **Lazy client initialization** — API clients connect on first tool call, not at import. This lets the MCP server start and register tools cleanly, even if env vars aren't set yet.
- **Kept everything else simple** — no LangChain, no caching layers, no retry logic. Clean Python with direct API calls.

## Files

```
rag_core.py      # Core RAG: chunking, embedding, retrieval, generation
mcp_server.py    # MCP server (thin wrapper around rag_core)
ingest.py        # One-time: chunk → embed → store → create vector index
faqs/            # FAQ markdown corpus
```
