from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"

documents = [
    "Delhi is the capital of India",
    "France is the capital of Spain",
    "Kolkata is the capital of WB",
]

vector = embedding.embed_documents(documents)

# vector = embedding.embed_query(text)
print(str(vector))