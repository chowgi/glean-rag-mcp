# FAQ RAG + MCP Tool

A retrieval-augmented generation (RAG) system that answers questions from FAQ markdown files, exposed as an MCP tool.

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | VoyageAI `voyage-3-lite` (512 dims) | Purpose-built for retrieval; outperforms OpenAI ada-002 on search benchmarks |
| **Vector Store** | MongoDB with `$vectorSearch` | Production-grade persistence & scalability vs in-memory numpy |
| **LLM** | OpenAI `gpt-4o-mini` | Cost-efficient, fast, plenty capable for FAQ Q&A |
| **MCP Transport** | stdio | Simple, reliable, standard for local MCP tools |

## Setup

### 1. Install dependencies

```bash
cd solution/
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**On Mac (local MongoDB):**
```bash
VOYAGE_API_KEY=your-key
OPENAI_API_KEY=your-key
# MONGODB_URI defaults to mongodb://localhost:27017/?directConnection=true
```

**In Docker container:**
```bash
VOYAGE_API_KEY=your-key
OPENAI_API_KEY=your-key
MONGODB_URI=mongodb://admin:password123@172.17.0.1:27017/?authSource=admin&directConnection=true
```

### 3. Start MongoDB

**Mac:**
```bash
brew services start mongodb-community
# Or: mongod --dbpath /path/to/data
```

**Docker (already running in container setup).**

### 4. Ingest FAQ files

```bash
python ingest.py
```

This chunks the FAQ markdown files, embeds them with VoyageAI, stores in MongoDB, and creates the vector search index.

### 5. Query (CLI test)

```bash
python rag_core.py
# Enter: "How do I reset my password?"
```

### 6. Run as MCP server

Configure your MCP client to spawn:

```json
{
  "mcpServers": {
    "faq-rag": {
      "command": "python",
      "args": ["/absolute/path/to/solution/mcp_server.py"],
      "env": {
        "VOYAGE_API_KEY": "your-key",
        "OPENAI_API_KEY": "your-key",
        "MONGODB_URI": "mongodb://localhost:27017/?directConnection=true"
      }
    }
  }
}
```

## MCP Tool

### `ask_faq`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | yes | — | Natural language question |
| `top_k` | number | no | 4 | Chunks to retrieve (1-10) |

**Response:**
```json
{
  "answer": "To reset your password, use the reset link on the login page (faq_auth.md). ...",
  "sources": ["faq_auth.md", "faq_sso.md"]
}
```

## File Structure

```
solution/
├── rag_core.py      # Core RAG: chunking, embedding, retrieval, generation
├── mcp_server.py    # MCP server (thin wrapper around rag_core)
├── ingest.py        # One-time ingestion script
├── faqs/            # FAQ markdown files
├── requirements.txt
├── .env.example
└── README.md
```

## Design Decisions

- **Simplicity over over-engineering**: No LangChain, no complex chunking strategies, no caching layers. Just clean Python with direct API calls.
- **Separation of concerns**: Ingestion (`ingest.py`) is separate from query time (`rag_core.py`). No re-embedding on every server start.
- **Fail fast**: Missing API keys raise immediately at import time, not at query time.
- **Configurable**: All settings via environment variables with sensible defaults.
