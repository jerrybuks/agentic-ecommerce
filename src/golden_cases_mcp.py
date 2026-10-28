"""Read-only stdio MCP server for Shoplytic's golden evaluation cases."""

from mcp.server import MCPServer

try:
    from .golden_loader import list_golden_cases as _list_golden_cases
except ImportError:  # Supports launching this file directly via the Inspector.
    from golden_loader import list_golden_cases as _list_golden_cases

server = MCPServer("shoplytic-golden-cases")


@server.tool()
def list_golden_cases(category: str | None = None) -> list[dict]:
    """List fixed Shoplytic golden cases, optionally by exact category."""
    return _list_golden_cases(category)


if __name__ == "__main__":
    server.run(transport="stdio")
