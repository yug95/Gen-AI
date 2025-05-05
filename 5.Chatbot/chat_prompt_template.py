from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# Define the system message, this is not working
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content='You are a helpful {domain} expert'),
#     HumanMessage(content='Explain me in simple term, {topic}'),
# ])

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain me in simple term, {topic}')
])

prompt = chat_template.invoke({'domain': 'Cricket', 'topic': 'LBW'})

print(prompt)

result = model.invoke(prompt)
print(result.content)
