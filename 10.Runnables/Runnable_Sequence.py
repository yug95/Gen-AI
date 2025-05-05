from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# Create a prompt template
prompt = PromptTemplate(
    template="Write a Joke on about {topic}",
    input_variables=[ "topic"]
)

prompt2 = PromptTemplate(
    template="explain the following {joke}",
    input_variables=["joke"]
)

# Create an LLM
llm = ChatOpenAI()

# Create an output parser
output_parser = StrOutputParser()

chain = RunnableSequence(prompt,llm,output_parser,prompt2,llm,output_parser)


# Run the chain
result = chain.invoke({"topic": "Python"})
print(result)  # This will print the random response from the LLM
