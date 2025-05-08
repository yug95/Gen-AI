from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

docs1 = Document(
    page_content="Virat Kohli is a top-order batsman known for his aggressive play and consistent performances. He has captained India across formats and is one of the highest run-scorers in modern cricket.",
    metadata={"team": "India","role": "Batsman"}
)

docs2 = Document(
    page_content="Ben Stokes is an explosive all-rounder renowned for match-winning performances, including a heroic knock in the 2019 World Cup Final and Ashes series",
    metadata={"team": "England","role": "All-rounder"}
)
docs3 = Document(
    page_content="Babar Azam is a stylish right-handed batsman who leads Pakistan in all formats. Known for his elegance and consistency, he's one of the top-ranked batters globally.",
    metadata={"team": "Pakistan","role": "Batsman"}
)

docs4 = Document(
    page_content="Kane Williamson is a technically sound batsman and the captain of New Zealand. He is known for his calm demeanor and ability to anchor innings.",
    metadata={"team": "New Zealand","role": "Batsman"}
)
docs5 = Document(
    page_content="Steve Smith is an unorthodox batsman known for his unique technique and ability to score runs in all conditions. He has been a key player for Australia in Test cricket.",
    metadata={"team": "Australia","role": "Batsman"}
)


docs = [docs1, docs2, docs3, docs4, docs5]
print(docs[0])

print(Chroma)



# # Initialize the OpenAI embeddings
vectorstore = Chroma(
                embedding_function=OpenAIEmbeddings(), 
                persist_directory="test_db" 
            )

# vectorstore = Chroma(
#     embedding_function=OpenAIEmbeddings(), 
#     persist_directory="My_chroma_db", 
#     collection_name="cricketers"
# )

# persist_directory = "My_chroma_db"

# # STEP 4: Create and store the vector store
# vectorstore = Chroma.from_documents(documents=docs,
#                                     embedding=OpenAIEmbeddings(),
#                                     persist_directory=persist_directory)

# vectorstore.persist()  # Save to disk
# print("ChromaDB initialized and persisted.")

# # STEP 5: Load vectorstore again (if needed)
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())