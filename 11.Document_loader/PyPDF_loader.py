from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

loader = PyPDFLoader("/Users/yogeshagrawal/Desktop/Gen AI/11.Document_loader/Receipt.pdf")

docs = loader.load()

# print(docs)

print(docs[0].page_content)
print(docs[0].metadata)