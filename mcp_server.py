"""MCP server: exposes ask_faq tool over stdio for Cursor/other MCP clients."""

import sys, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # so server has API keys when run by MCP host
# Log errors to file; otherwise MCP host may hide stderr
sys.stderr = open(Path(__file__).parent / "mcp_debug.log", "a")

from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core

mcp = FastMCP("faq-rag")

@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> str:
    """Answer from FAQ corpus; cite sources. Returns JSON string (answer + sources) or error."""
    try:
        # Clamp top_k to 1–10
        result = ask_faq_core(question.strip(), top_k=max(1, min(top_k or 4, 10)))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run(transport="stdio")  # Cursor talks to this process via stdin/stdout
