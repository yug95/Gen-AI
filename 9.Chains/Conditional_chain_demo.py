from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Feedback(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="give the Sentiment of the feedback")

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(template=" Classfiy the following feedback into positive or negative \n {feedback} \n {format_instructions}",
input_variables=["feedback"],
partial_variables={"format_instructions": parser2.get_format_instructions()}
)


classifer_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(template="write an appropiate response to this positive feedback \n {feedback}",
                        input_variables=["feedback"]
)

prompt3 = PromptTemplate(template="write an appropiate response to this negative feedback \n {feedback}",
                        input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment")
)

final_chain  = classifer_chain | branch_chain

result = final_chain.invoke({"feedback": "The product is terrible"})

print(result)

print(final_chain.get_graph().print_ascii())