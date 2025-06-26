from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(city: str) -> str:
    """Returns a mock weather report for the given city."""
    # In a real application, you would fetch this data from a weather API.
    return f"The weather in {city} is sunny with a high of 25°C."


if __name__ == "__main__":
    mcp.run(transport='streamable-http')