from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

class CustomState(TypedDict):
    text : str


def Node_A(state: CustomState):
    return Command(
        goto="Node_B",
        update={
        "text": state["text"] + "a"
    }
    )

def Node_B(state: CustomState):
    return Command(
        goto="Node_C",
        update={
        "text": state["text"] + "b"
    }
    )

def Node_C(state: CustomState):
    return Command(
        goto=END,
        update={
        "text": state["text"] + "c"
    }
    )

graph = StateGraph(CustomState)
graph.add_node("Node_A", Node_A)
graph.add_node("Node_B", Node_B)
graph.add_node("Node_C", Node_C)
graph.set_entry_point("Node_A")
app = graph.compile()
# Example usage
state = CustomState(text="")
response = app.invoke(state)
print(response)  # Should print "abc"