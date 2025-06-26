from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtracts the second integer from the first."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divides the first integer by the second."""
    return a / b

if __name__ == "__main__":
    mcp.run(transport='stdio')
