from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import ServiceContext, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv

load_dotenv()

docs = SimpleDirectoryReader("/Users/yogeshagrawal/Desktop/Gen AI/24.Llama_index/data").load_data()


# Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

# splitter = SentenceSplitter(
#     chunk_size=1024,
#     chunk_overlap=20,
# )
# nodes = splitter.get_nodes_from_documents(docs)


# per-index
index = VectorStoreIndex.from_documents(
    docs,
    transformations=[SentenceSplitter(chunk_size=500, chunk_overlap=50)]
)

query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of India?")
print(response)
