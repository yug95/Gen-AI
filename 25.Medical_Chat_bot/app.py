from flask import Flask, render_template, jsonify, request
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.helper import data_loader, text_chunk
from src.retriever import retrieve_and_rerank
from src.prompt import get_prompt
from src.memory import get_session_history
import numpy as np

load_dotenv()

# loader = DirectoryLoader("/Users/yogeshagrawal/Desktop/Gen AI/25.Medical_Chat_bot/Data", glob="*.pdf", loader_cls=PyPDFLoader)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
# texts = text_splitter.split_documents(docs)

# docs = data_loader()

# texts = text_chunk(docs)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# vectorstore = FAISS.from_documents(texts, embeddings)

# llm_model = ChatOpenAI()

# parser = StrOutputParser()

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})
# compressor = LLMChainExtractor.from_llm(llm_model)

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever  # your existing retriever
# )

# source = []
# def format_docs(retrieved_docs):
#     metadata = []
#     for docs in retrieved_docs:
#         source = docs.metadata['source']
#         page_label = docs.metadata['page_label']
#         metadata.append({"source":source,"page_label": page_label})

#     print(metadata)
    
#     return "\n\n".join([doc.page_content for doc in retrieved_docs])


# def dot_product(a, b):
#     return np.dot(a, b)

# def rerank_openai(query: str, docs, embeddings_model, top_k=5):
#     query_embedding = embeddings_model.embed_query(query)
    
#     doc_embeddings = [embeddings_model.embed_query(doc.page_content) for doc in docs]
#     scores = [dot_product(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    
#     ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
#     print(ranked_docs)
#     return [doc for _, doc in ranked_docs[:top_k]]

# def retrieve_and_rerank(inputs):
#     raw_docs = compression_retriever.invoke(inputs["question"])
#     reranked = rerank_openai(inputs["question"], raw_docs,embeddings)
#     return reranked


reranked_retriever = RunnableLambda(retrieve_and_rerank)

system_prompt = (
    """
    You are a helpful medical assistant for question answering task.use the following pieces of context to answer the answer.If you don't know the answer, just say "I don't know".
    \n\n
    {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{chat_history}"),
        ("human", "{input}")
    ]
)

prompt = RunnableLambda(get_prompt)


# class TruncatedChatMessageHistory(ChatMessageHistory):
#     def append(self, message):
#         super().append(message)
#         # Keep only the last 5 messages
#         if len(self.messages) > 5:
#             self.messages = self.messages[-5:]



# # Dictionary to hold memory per session
# message_histories = {}


# def get_session_history(session_id):
#     if session_id not in message_histories:
#         message_histories[session_id] = TruncatedChatMessageHistory()
#     return message_histories[session_id]




source = []
def format_docs(retrieved_docs):
    metadata = []
    for docs in retrieved_docs:
        source = docs.metadata['source']
        page_label = docs.metadata['page_label']
        metadata.append({"source":source,"page_label": page_label})

    print(metadata)
    
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

parallel_chain = RunnableParallel({
   "context": reranked_retriever | RunnableLambda(format_docs),
    "input": RunnableLambda(lambda x: x["question"]),
    "chat_history": RunnableLambda(lambda x: x.get("chat_history", ""))
})

llm_model = ChatOpenAI()
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm_model | parser


# Wrap the chain with memory management
rag_with_memory = RunnableWithMessageHistory(
    main_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def ask():
    input = request.form['msg']
    # print(input)
    # print(message_histories)
    response = rag_with_memory.invoke({"question": input},
                           config={"configurable": {"session_id": "user-001"}
                        })
    # response = main_chain.invoke(input)
    print(response)
    return response

if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 8081,debug=True)
