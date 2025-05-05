from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Create a prompt template
prompt = PromptTemplate(
    template="Write a Joke on about {topic}",
    input_variables=["topic"]
)

llms = ChatOpenAI()
output_parser = StrOutputParser()

joke_gen_parser = RunnableSequence(prompt, llms, output_parser)

def word_count(joke):
    return len(joke.split())

Parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'Word_count': RunnableLambda(word_count)
})

# Create a sequence runnable
final_chain = RunnableSequence(joke_gen_parser, Parallel_chain)
result = final_chain.invoke({"topic": "Python"})
print(result)  # This will print the random response from the LLM
print(final_chain.get_graph().print_ascii())