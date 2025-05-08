from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

######### Indexing 
#step1 : Load Youtube video transcript through Youtube API

video_id = "Gfr50f6ZBvo"  # Replace with your YouTube video ID
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
    # print("Transcript loaded successfully.", transcript_list)

    #Create a text file with the transcript
    transcript = "".join(chunk['text'] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("Transcript not available for this video.")




#step2 : Split the transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))
# print("Chunks created successfully.", chunks[0])



# step3 : Create embeddings for the chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# step4 : Create a FAISS vector store from the chunks and embeddings
vectorstore = FAISS.from_documents(chunks, embeddings)

# print("Vector store created successfully.",vectorstore.index_to_docstore_id)
# print(vectorstore.get_by_ids([vectorstore.index_to_docstore_id[0]]))


# step 2 : Retrieving the most relevant chunks

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# print("Retriever created successfully.", retriever)
# print(retriever.invoke("what is deepmind ?"))


# step 3 : Create a prompt template for the LLM Augmentation

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Answer only from the provided transcript context. if context is insufficient just say don't know.
    {context}
    Question: {question}:""",
)

Question = "What is DeepMind?"
# retrieved_docs = retriever.invoke(Question)

# print("Retrieved documents successfully.", retrieved_docs)

# context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# print("Context retrieved successfully.", context)

# final_prompt = prompt_template.invoke({"context": context, "question": Question})
# print("Final prompt created successfully.", final_prompt) 


# step 4 : Create a ChatOpenAI instance and get the answer
llm = ChatOpenAI()

# llm_response = llm.invoke(final_prompt)
# print("LLM response successfully.", llm_response)
# step 5 : Print the answer


# Creating everything using chains

from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def format_docs(retrieved_docs):
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


parallel_chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'context': retriever | RunnableLambda(format_docs)
})

# print(parallel_chain.invoke('who is demis hassabis ?'))

parser = StrOutputParser()

main_chain = parallel_chain | prompt_template | llm | parser

print(main_chain.invoke("What is the mission of DeepMind ?"))