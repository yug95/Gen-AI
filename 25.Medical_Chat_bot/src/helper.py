from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def data_loader():
    loader = DirectoryLoader("/Users/yogeshagrawal/Desktop/Gen AI/25.Medical_Chat_bot/Data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def text_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(docs)
    return texts


