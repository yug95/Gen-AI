from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template="Generate 5 intrested fact about : {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = template1 | model | parser

result = chain.invoke({"topic": "Blackhole"})

print(result)

chain.get_graph().print_ascii()


