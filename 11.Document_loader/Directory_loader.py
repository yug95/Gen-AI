from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

DirectoryLoader = DirectoryLoader(
    "/Users/yogeshagrawal/Desktop/Gen AI/11.Document_loader/pdf_directory",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = DirectoryLoader.load() # DirectoryLoader.lazy_load()
print(docs[1].page_content)
print(docs[1].metadata)