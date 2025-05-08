from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.retrievers import MMRRetriever
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Initialize the OpenAI embeddings
FAISS_vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

retriever = FAISS_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult":0.5}

)

query = "What is LangChain?"
result = retriever.invoke(query)
print("Result:", result)