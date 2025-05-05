from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

# to store the chat history and make sure the model remembers the context
# chat_list = []
chat_list = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("You:")
    chat_list.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(user_input)
    # chat_list.append(result.content)
    chat_list.append(AIMessage(content=result.content))
    print("Chatbot:", result.content)

print("Chat History:",chat_list)