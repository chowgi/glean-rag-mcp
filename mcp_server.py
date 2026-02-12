"""
MCP server exposing the ask_faq tool via stdio transport.

Intentionally thin — all RAG logic lives in rag_core.py.
This file just wires up the MCP interface.
"""

import os
from typing import Dict

# Load .env file if present — handy for local dev on Mac
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core

# Fail fast — surface config problems at startup, not at query time
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")
if not os.getenv("VOYAGE_API_KEY"):
    raise RuntimeError("VOYAGE_API_KEY is not set")

# Create the MCP server instance
mcp = FastMCP("faq-rag")


@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> Dict[str, object]:
    """
    Answer a question from the FAQ corpus and cite at least two source files.

    This is the MCP tool that clients call. It validates inputs and
    delegates to ask_faq_core() which does the actual RAG pipeline.

    Args:
        question: Natural language question to answer from the FAQ knowledge base.
        top_k: Number of chunks to retrieve (1-10, default 4).

    Returns:
        {"answer": "...", "sources": ["file1.md", "file2.md"]}
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("`question` is required")

    # Clamp to valid range
    top_k = max(1, min(top_k or 4, 10))

    return ask_faq_core(q, top_k=top_k)


if __name__ == "__main__":
    # stdio transport: the MCP client launches this as a subprocess
    # and communicates via stdin/stdout JSON-RPC messages
    mcp.run(transport="stdio")
