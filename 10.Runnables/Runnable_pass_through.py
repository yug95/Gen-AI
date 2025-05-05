from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Create a prompt template
prompt = PromptTemplate(
    template="Write a Joke on about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain the following {joke}",
    input_variables=["joke"]
)

# Create an LLM
llm = ChatOpenAI()

# Create an output parser
output_parser = StrOutputParser()

# Create a passthrough runnable
passthrough = RunnablePassthrough()

# Create a sequence runnable
Joke_gen_parser = RunnableSequence(prompt, llm, output_parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain': RunnableSequence(prompt2, llm, output_parser)
})

# Create a sequence runnable
final_chain = RunnableSequence(Joke_gen_parser,parallel_chain)

print("Final Result")
result = final_chain.invoke({"topic": "Python"})
print(result)  # This will print the random response from the LLM

print(final_chain.get_graph().print_ascii())