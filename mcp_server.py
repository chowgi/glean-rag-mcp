"""
MCP server exposing ask_faq tool via stdio transport.
All logic lives in rag_core.py — this is just the MCP wrapper.
"""

import sys, os, json
from pathlib import Path

# Ensure .env is loaded relative to this file (not the MCP client's cwd)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Redirect stderr to a log file so MCP client doesn't swallow errors silently
log_path = Path(__file__).parent / "mcp_debug.log"
sys.stderr = open(log_path, "a")

from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core

mcp = FastMCP("faq-rag")

@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> str:
    """Answer a question from the FAQ corpus and cite at least two source files.

    Args:
        question: Natural language question about company FAQs
        top_k: Number of chunks to retrieve (1-10, default 4)
    """
    # MCP tools should return a string — serialise the result as JSON
    result = ask_faq_core(question.strip(), top_k=max(1, min(top_k or 4, 10)))
    return json.dumps(result)

if __name__ == "__main__":
    mcp.run(transport="stdio")
