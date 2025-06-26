from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():

    client = MultiServerMCPClient(
        {
            "MathServer" : {
                "command" : "python",
                "args" : ["/Users/yogeshagrawal/Desktop/Gen AI/30.MCP_Demo/mathserver.py"],
                "transport" : "stdio"
            },

            "WeatherServer" : {
                "url" : "http://127.0.0.1:8000/mcp",
                "transport" : "streamable_http"
            }

        }
)

    tools = await client.get_tools()

    # print(tools)

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(
        model=model,
        tools=tools
    )

    response = await agent.ainvoke({
        'messages':[{'role': 'user', 'content': 'weather in New York'}],
    })

    print(response['messages'][-1].content)

asyncio.run(main())
