from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model='gpt-4',temperature=1.5)
model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_tokens=10)

result = model.invoke("who is pm of india")
print(result.content)