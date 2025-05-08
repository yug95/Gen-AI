from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=5,lang="en")

query = "Histroy of mughal empire ?"

docs = retriever.invoke(query)

print("Result:",len(docs))
print("Result:",docs[0].page_content)
print("Result:",docs[0].metadata)