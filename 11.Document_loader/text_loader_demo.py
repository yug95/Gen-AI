from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("/Users/yogeshagrawal/Desktop/Gen AI/11.Document_loader/cricket_poem.txt", encoding="utf-8")

docs = loader.load()

# print(docs)

# print(type(docs))

# print(docs[0])

# print(docs[0].page_content)
# print(docs[0].metadata)

model = ChatOpenAI()
output_parser = StrOutputParser()
prompt = PromptTemplate(
    template="Write a summary of the following {poem}",
    input_variables=["poem"]
)

chain = prompt | model | output_parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)  # This will print the random response from the LLM