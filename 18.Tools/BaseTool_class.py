from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

class MultiplySchema(BaseModel):
    x: int = Field(required=True, description="First number to multiply")
    y: int = Field(required=True, description="Second number to multiply")


class MultiplyTool(BaseTool):
    name : str = "multiply"
    description : str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplySchema

    def _run(self, x: int, y: int) -> int:
        """Run the tool."""
        return multiply(x, y)
    

multiply_tool = MultiplyTool()
multiply_result = multiply_tool.invoke({"x": 2, "y": 3})
print("Multiply result successfully.", multiply_result) 
print("Multiply tool name successfully.", multiply_tool.name)
print("Multiply tool description successfully.", multiply_tool.description)
print("Multiply tool args successfully.", multiply_tool.args)