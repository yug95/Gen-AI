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
import numpy as np






def dot_product(a, b):
    return np.dot(a, b)

def rerank_openai(query: str, docs, embeddings_model, top_k=5):
    query_embedding = embeddings_model.embed_query(query)
    
    doc_embeddings = [embeddings_model.embed_query(doc.page_content) for doc in docs]
    scores = [dot_product(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    
    ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    print(ranked_docs)
    return [doc for _, doc in ranked_docs[:top_k]]

def retrieve_and_rerank(inputs):

    docs = data_loader()
    texts = text_chunk(docs)
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever  # your existing retriever
    )

    raw_docs = compression_retriever.invoke(inputs["question"])
    reranked = rerank_openai(inputs["question"], raw_docs,embeddings)
    return reranked
