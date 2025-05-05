from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a summary of the following {text}",
    input_variables=["text"]
)

model = ChatOpenAI()
output_parser = StrOutputParser()


report_gen = RunnableSequence(prompt1, model, output_parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>50, RunnableSequence(prompt2, model, output_parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen, branch_chain)
result = final_chain.invoke({"topic": "Python"})
print(result)  # This will print the random response from the LLM
print(final_chain.get_graph().print_ascii())