from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

documents = [
    "Delhi is the capital of India",
    "France is the capital of Spain",
    "Kolkata is the capital of WB",
]

# result = embeddings.embed_query("Delhi is the capital of India")
result = embeddings.embed_documents(documents)

print(str(result))