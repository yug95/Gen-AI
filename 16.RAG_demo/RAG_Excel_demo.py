
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()

st.title("📊 Sales Data Q&A (RAG with Excel Upload)")

uploaded_file = st.file_uploader("Upload your sales Excel file", type=["xlsx"])

if uploaded_file:
    # Read Excel
    df = pd.read_excel(uploaded_file)
    st.write("Sample Data:", df.head())

    # Convert DataFrame to plain text (you can improve chunking later)
    data_text = df.to_csv(index=False)

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([data_text])

    # Embed & index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define LLM
    llm = ChatOpenAI(temperature=0)

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Ask question
    user_query = st.text_input("Ask a question about the sales data:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        st.subheader("Answer:")
        st.write(result["result"])

        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)
