from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template1 = PromptTemplate(template="Generate the name, age and city of a fictional {place} person \n {format_instructions}",
                            input_variables=["place"],
                            partial_variables={"format_instructions": parser.get_format_instructions()}
                    )

# prompt = template1.invoke({"place": "Spain"})

chain = template1 | model | parser

result = chain.invoke({"place": "Spain"})

print(result)