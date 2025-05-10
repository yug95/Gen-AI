from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

llm = ChatOpenAI()


llm_tool = llm.bind_tools([multiply])

# print(llm_tool.invoke("how are you ?"))

# print(llm_tool.invoke("can you multiply 2 and 3"))  # 6
# print(llm_tool.invoke("can you multiply 2 and 3").tool_calls)  # 6
# print(llm_tool.invoke("can you multiply 2 and 3").tool_calls[0])  # multiply


# print(multiply.invoke(llm_tool.invoke("can you multiply 2 and 3").tool_calls[0]))


# to make it more sophisticated, we can use the tool to call the llm_tool

query = HumanMessage("can you multiply 2 and 3")
messages_list = [query]

result = llm_tool.invoke(messages_list)

messages_list.append(result)

tool_result = multiply.invoke(result.tool_calls[0])

messages_list.append(tool_result)

print(messages_list)

print(llm_tool.invoke(messages_list))