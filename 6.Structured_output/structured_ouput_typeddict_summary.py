from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    # summary: str
    # sentiment: str
    summary: Annotated[str,"Summary of the review"]
    sentiment: Annotated[str,"Sentiment of the review either negative or positive or neutral"]
    pros: Annotated[Optional[list[str]], "Pros of the product"] # type: ignore if pros is not required

structured_output = model.with_structured_output(Review)

result = structured_output.invoke("Summarize the following text and provide a sentiment: 'I love programming in Python. It's so versatile and powerful!' pros : it is very robust and easy to learn. cons : it is not very fast.'")

print(result)
print(result['summary'])
print(result['sentiment'])
print(result.keys())