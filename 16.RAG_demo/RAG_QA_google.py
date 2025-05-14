from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("/Users/yogeshagrawal/Desktop/Gen AI/16.RAG_demo/attention_is_all_you_need.pdf")
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm_model = ChatOpenAI()


system_prompt = (
    """
    You are a assistant for question answering task.use the following pieces of context to answer the answer.If you don't know the answer, just say "I don't know".Write answer in a concise manner and within 3-4 lines.
    \n\n
    {context}
    """
)

# prompt = ChatPromptTemplate.from_messages([
#     ('system', 'You are a helpful Question answering expert, answer based on the {context}'),
#     ('human', 'Explain me in simple term, {question}')
# ])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()

# print(prompt.invoke({"context": "This is a test context", "question": "What is LLaMA 2?"}))

# Define a wrapper that prepares the inputs for your prompt
# --- Define how to prepare inputs for the prompt ---

# Define a wrapper that prepares the inputs for your prompt
# def format_input(inputs):
#     return {
#         "context": "\n\n".join([doc.page_content for doc in inputs["input_documents"]]),
#         "question": inputs["question"]
#     }

#option 1
combine_docs_chain = create_stuff_documents_chain(llm_model, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
# Create your chain


result = rag_chain.invoke({'input':'what is Narendra ?'})
print(result['answer'])


#option 2

# def format_docs(retrieved_docs):
#     return "\n\n".join([doc.page_content for doc in retrieved_docs])


# parallel_chain = RunnableParallel({
#     'question': RunnablePassthrough(),
#     'context': retriever | RunnableLambda(format_docs)
# })

# result = parallel_chain.invoke('what is LLama 2 ?')

# main_chain = parallel_chain | prompt | llm_model | parser

# # # # --- Run the chain ---
# print(main_chain.invoke("What is LLaMA 2?"))