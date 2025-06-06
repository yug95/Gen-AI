from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from chains import generation_chain, reflection_chain

from langgraph.graph import END, MessageGraph
from dotenv import load_dotenv

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

#intiatiate the message graph
graph = MessageGraph()

# Define the generation node``
def generate_node(state):
    return generation_chain.invoke({
        "messages":state
    })

# define the reflection node
def reflect_node(state):
    # can pass reflction output as human reponse just to feel like AI and Human are talking.
    response = reflection_chain.invoke({
        "messages": state
    })
    return [HumanMessage(content=response.content)]

    # this is the same as above, but we return the response as a system message but doesn't matter in terms of functionality
    # return reflection_chain.invoke({
    #     "messages":state
    # })


# add nodes to the graph

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)

def should_continue(state):
    # check if the last message is a human message
    if (len(state) > 4):
        return END
    return REFLECT

# add the conditional edges

# graph.add_node("ShouldContinue", should_continue) # why it is not a node ?

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation."))

print("Response:", response)

