"""
MCP server exposing ask_faq tool via stdio transport.
All logic lives in rag_core.py — this is just the MCP wrapper.
"""

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core

mcp = FastMCP("faq-rag")

@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> dict:
    """Answer a question from the FAQ corpus and cite at least two source files."""
    return ask_faq_core(question.strip(), top_k=max(1, min(top_k or 4, 10)))

if __name__ == "__main__":
    mcp.run(transport="stdio")
