from langchain_core.tools import tool

@tool
def multiply(x:int, y:int) -> int:
    """Multiply two numbers."""
    return x * y


result = multiply.invoke({"x":2, "y":3})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema())