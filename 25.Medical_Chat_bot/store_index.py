from src.helper import data_loader, text_chunk
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from pro

from dotenv import load_dotenv

load_dotenv()

docs = data_loader()
texts = text_chunk(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the vector store
vectorstore = FAISS.from_documents(texts, embeddings)

