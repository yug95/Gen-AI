from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_history = []

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful Assistant'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])


with open('/Users/yogeshagrawal/Desktop/Gen AI/5.Chatbot/chat_history.txt') as f:
    chat_history.extend(f.readlines())


prompt = chat_template.invoke({ 'chat_history': chat_history, 'query': 'Where is my refund ?'})

print(prompt)