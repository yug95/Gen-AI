import os
# from langchain.vectorstores import Pinecone
# import pinecone
from langchain_openai.embeddings import OpenAIEmbeddings



# PINECORE_API_KEY = os.environ.get("PINECONE_API_KEY","pcsk_3gZKPR_NGf2E8oeyEc5xvBNqJZGToMDx8Vu8aVWEfQZ3zYkh67kmVCsun2pM5mD53aGR8G")
# PINECORE_API_ENV = os.environ.get("PINECONE_API_ENV","AWS")

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="pcsk_3gZKPR_NGf2E8oeyEc5xvBNqJZGToMDx8Vu8aVWEfQZ3zYkh67kmVCsun2pM5mD53aGR8G")

index_name = "testpinedb"

# pc.create_index(
#     name=index_name,
#     dimension=1536, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

# pinecone.init(
#     api_key=PINECORE_API_KEY,
#     environment=PINECORE_API_ENV
# )
# index_name = "testpinedb"

docs = pc.from_texts(
    texts=["Hello world", "Goodbye world"],
    embedding=OpenAIEmbeddings(),
    index_name=index_name
)

