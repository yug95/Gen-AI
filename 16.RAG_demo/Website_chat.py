from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

import os
os.environ["SSL_CERT_FILE"] = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/certifi/cacert.pem"

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

load_dotenv()

Urls = [
    'https://blog.gopenai.com/paper-review-llama-2-open-foundation-and-fine-tuned-chat-models-23e539522acb',
    'https://www.databricks.com/blog/mpt-7b',
    'https://lmsys.org/blog/2025-05-05-large-scale-ep/',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]

loader = UnstructuredURLLoader(urls=Urls)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text_chunks = text_splitter.split_documents(documents)
# print(text_chunks[0])

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()
llm_model = ChatOpenAI()



# chain = load_qa_with_sources_chain( 
#         llm = llm_model,
#         chain_type='stuff',
#     )

query = 'How good is Vicuna?'
# relevant_docs = retriever.get_relevant_documents(query)

chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


result = chain.invoke({'query':query})
print(result['result'])

