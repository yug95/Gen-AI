from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# model = ChatAnthropic(model='claude-2', temperature=1.5)
model = ChatAnthropic(model='claude-3.5-sonnet-20241022', temperature=0.7, max_completion_tokens=10)

result = model.invoke("who is pm of india")
print(result.content)
