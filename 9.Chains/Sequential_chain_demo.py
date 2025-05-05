from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

#prompt 1
template1 = PromptTemplate(
    template="Generate a detailed report on: {topic}",
    input_variables=["topic"],
)

# prompt 2
template2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following \n : {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

print(chain.invoke({"topic": "Blackhole"}))

chain.get_graph().print_ascii()