from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#using local model
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#using end points
embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is a cricketer and former captain of the Indian national team.",
    "MS Dhoni named thala and captain of csk",
    "Jasprit Bumrah is a fast bowler and a key player for India.",
    "Hitman Rohit sharma scored fasted 200",
    "Australia is a country and continent surrounded by the Indian and Pacific oceans."
]

document_embedding = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(" who is Virat Kohli ?")

query_embedding = embedding.embed_query(" what is 2+2 ?")

# Calculate cosine similarity
similarity_scores = cosine_similarity([query_embedding], document_embedding)

print(similarity_scores)
# Get the index of the most similar document
most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]
most_similar_score = similarity_scores[0][most_similar_index]
print(f"Most similar document: {most_similar_document}")
print(f"Similarity score: {most_similar_score}")


