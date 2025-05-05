from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv


load_dotenv()

# Create a prompt template
prompt = PromptTemplate(
    template="Write a Tweet on about {topic}",
    input_variables=[ "topic"]
)
prompt2 = PromptTemplate(
    template="write a linkeding post about {topic}",
    input_variables=["topic"]
)

# Create an LLM
llm = ChatOpenAI()
# Create an output parser
output_parser = StrOutputParser()
# Create a parallel runnable

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt,llm,output_parser),
    'linkeding':RunnableSequence(prompt2,llm,output_parser)
})

result = parallel_chain.invoke({"topic": "Python"})
print(result)  # This will print the random response from the LLM