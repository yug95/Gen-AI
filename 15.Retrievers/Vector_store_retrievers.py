from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

docs1 = Document(page_content="Virat Kohli is a top-order batsman known for his aggressive play and consistent performances.")
docs2 = Document(page_content="Ben Stokes is an explosive all-rounder renowned for match-winning performances")
docs3 = Document(page_content="Babar Azam is a stylish right-handed batsman who leads Pakistan in all formats.")
docs4 = Document(page_content="Kane Williamson is a technically sound batsman and the captain of New Zealand.")
docs5 = Document(page_content="Steve Smith is an unorthodox batsman known for his unique technique and ability to score runs in all conditions.")
docs = [docs1, docs2, docs3, docs4, docs5]

embedding = OpenAIEmbeddings()
# Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

#create a vector store retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "Who is the captain of New Zealand?"
result = retriever.invoke(query)

print("Result:", result)


