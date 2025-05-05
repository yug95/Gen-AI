from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    summary: str = Field(description="Summary of the review")
    sentiment: str = Field(description="Sentiment of the review either negative or positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Pros of the product") # type: ignore if pros is not required
    cons: Optional[list[str]] = Field(default=None, description="Cons of the product") # type: ignore if cons is not required

structured_output = model.with_structured_output(Review)

result = structured_output.invoke("Summarize the following text and provide a sentiment: 'I love programming in Python. It's so versatile and powerful!' pros : it is very robust and easy to learn. cons : it is not very fast.'")

print(result)
# print(result['summary'])
# print(result['sentiment'])
# print(result.keys())