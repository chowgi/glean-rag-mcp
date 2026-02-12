# FAQ RAG + MCP Tool

A RAG system that answers questions from FAQ markdown files, exposed as an MCP tool.

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | VoyageAI `voyage-3-lite` (512d) | Purpose-built for retrieval; outperforms ada-002 on search benchmarks |
| **Vector Store** | MongoDB `$vectorSearch` | Persistent, scalable, production-realistic vs in-memory numpy |
| **LLM** | OpenAI `gpt-4o-mini` | Cost-efficient, fast, plenty capable for FAQ Q&A |
| **MCP** | stdio transport | Standard for local MCP tools |

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

python ingest.py        # One-time: chunk, embed, store in MongoDB
python rag_core.py      # CLI test
```

## MCP Server (Claude Desktop)

Add to your MCP client config:

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

### `ask_faq` tool

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `question` | string | yes | — |
| `top_k` | number | no | 4 (1-10) |

Returns `{"answer": "...", "sources": ["file1.md", "file2.md"]}`

## Design Decisions

- **No LangChain or frameworks** — clean Python with direct API calls. Easier to read and debug.
- **Ingestion separate from query** — `ingest.py` runs once; no re-embedding on every server start.
- **Fail fast** — missing API keys raise at import time, not at query time.
- **All config via env vars** — `MONGODB_URI`, `EMBED_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`, `TOP_K_DEFAULT`.

## Files

```
rag_core.py      # Core RAG: chunking, embedding, retrieval, generation
mcp_server.py    # MCP server (thin wrapper)
ingest.py        # One-time ingestion + vector index creation
faqs/            # FAQ markdown corpus
```
