from typing import Annotated, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langgraph.graph import StateGraph, END,add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
load_dotenv()

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


llm_model = ChatOpenAI()
search_tool = TavilySearchResults()
memory = MemorySaver()

tools = [search_tool]

llm_with_tools = llm_model.bind_tools(tools)


class Agent_state(TypedDict):
   messages : Annotated[list, add_messages]


async def model_node(state:Agent_state):
    result = await llm_with_tools.ainvoke(state['messages'])
    return {
        "messages":[result]
    }


async def tool_router(state:Agent_state):
    last_message = state["messages"][-1]
    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
async def tool_node(state:Agent_state):
    tool_calls = state["messages"][-1].tool_calls
    print(tool_calls)

    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            # Invoke the search tool with the arguments
            search_result = await search_tool.ainvoke(tool_args)
            # Create a message for the tool response
            tool_message = ToolMessage(
                content = str(search_result),
                tool_call_id = tool_id,
                tool_name = tool_name,
            )
    
            tool_messages.append(tool_message)
    
    return {
        'messages':tool_messages
    }

graph_builder = StateGraph(Agent_state)
graph_builder.add_node("model_node", model_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model_node")
graph_builder.add_conditional_edges("model_node", tool_router)
graph_builder.add_edge("tool_node", "model_node")

graph = graph_builder.compile(checkpointer=memory)

config = {
    "configurable":{
        "thread_id":2
    }
}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"]
)

async def generate_response(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
  
        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        } 

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

    async for event in events:
        if event['event'] == "on_chat_model_stream":
            chunk = event['data']['chunk']
            yield f"data: {{\"type\": \"content\", \"content\": \"{chunk.content}\"}}\n\n"
        #     if isinstance(chunk, AIMessage):
        #         yield f"data: {{\"type\": \"message\", \"content\": \"{chunk.content}\"}}\n\n"
        #     elif isinstance(chunk, ToolMessage):
        #         yield f"data: {{\"type\": \"tool\", \"content\": \"{chunk.content}\", \"tool_name\": \"{chunk.tool_name}\"}}\n\n"
        # elif event['event'] == "on_chat_model_end":
        #     yield "data: {\"type\": \"end\"}\n\n" 
    
@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    
    return StreamingResponse(generate_response(message, checkpoint_id), media_type="text/event-stream")

# response = await graph.ainvoke({"messages": ["when is the next spaceX Launch ?"]}, config=config)

# events = graph.astream_events(
#     {"messages":["when is the next spaceX Launch ?"]},
#     config=config,
# )

# async for event in events:
#     # print(event)
#     if event['event'] == "on_chat_model_stream":
#         print(event['data']['chunk'].content, end='')