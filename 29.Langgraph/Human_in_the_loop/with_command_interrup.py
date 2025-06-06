from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)
memory = MemorySaver()

class CustomState(TypedDict):
    text : str


def Node_A(state: CustomState):
    print("Node_A")
    return Command(
        goto="Node_B",
        update={
        "text": state["text"] + "a"
    }
    )

def Node_B(state: CustomState):
    print("Node_B")
    Human_reponse = interrupt("Do you want C or D? (C/D): ")

    if Human_reponse == "C":
        print("You chose C, continuing the flow.")
        return Command(
            goto="Node_C",
            update={
                "text": state["text"] + "b"
            }
        )
    else:
        print("You chose D, interrupting the flow.")
        return Command(
            goto="Node_D",
            update={
                "text": state["text"] + "b (interrupted)"
            }
        )

def Node_C(state: CustomState):
    return Command(
        goto=END,
        update={
        "text": state["text"] + "c"
    }
    )

def Node_D(state: CustomState):
    return Command(
        goto=END,
        update={
        "text": state["text"] + "d"
    }
    )

graph = StateGraph(CustomState)
graph.add_node("Node_A", Node_A)
graph.add_node("Node_B", Node_B)
graph.add_node("Node_C", Node_C)
graph.add_node("Node_D", Node_D)
graph.set_entry_point("Node_A")
app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": 1
}}
# Example usage
state = CustomState(text="")
response = app.invoke(state, config, stream_mode="updates")
print(response)  # Should print "abc"

second_result = app.invoke(Command(resume="D"),config=config, stream_mode="updates")
print(second_result)  # Should print "abc (interrupted)"