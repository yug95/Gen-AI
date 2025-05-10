from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

class MultiplyTool(BaseModel):
    x : int = Field( required= True, description="First number to multiply")
    y : int = Field( required= True, description="Second number to multiply")


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyTool,
)

multiply_result = multiply_tool.invoke({"x": 2, "y": 3})
print("Multiply result successfully.", multiply_result)
print("Multiply tool name successfully.", multiply_tool.name)
print("Multiply tool description successfully.", multiply_tool.description)
print("Multiply tool args successfully.", multiply_tool.args)
print("Multiply tool args schema successfully.", multiply_tool.args_schema.model_json_schema())