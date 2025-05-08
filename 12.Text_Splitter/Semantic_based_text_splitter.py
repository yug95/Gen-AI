from langchain_experimental.text_splitter import SemanticChunker # type: ignore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings


# Load a PDF file
loader = PyPDFLoader("/Users/yogeshagrawal/Desktop/Gen AI/12.Text_Splitter/sample_generative_ai.pdf")
docs = loader.load()
# Initialize the text splitter

openai_embeddings = OpenAIEmbeddings()

text_splitter = SemanticChunker(openai_embeddings, breakpoint_threshold_type="standard_deviation", 
                                breakpoint_threshold_amount=1)

# Split the text into chunks
chunks = text_splitter.split_documents(docs)
# Print the number of chunks and the first chunk    
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[1].page_content}")
print(f"First chunk metadata: {chunks[1].metadata}")