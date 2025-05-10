from langchain_core.tools import tool

@tool
def multiply(x:int, y:int) -> int:
    """Multiply two numbers."""
    return x * y

@tool
def add(x:int, y:int) -> int:
    """Add two numbers."""
    return x + y

class MathToolKit:
    def __init__(self):
        self.multiply_tool = multiply
        self.add_tool = add

    def get_tools(self):
        return [self.multiply_tool, self.add_tool]
    

# Example usage
toolkit = MathToolKit()
tools = toolkit.get_tools()
for tool in tools:
    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description}")
    print(f"Tool args: {tool.args}")
    print(f"Tool args schema: {tool.args_schema.model_json_schema()}")

print(tools[0].invoke({"x": 2, "y": 3}))  # Example invocation of the multiply tool