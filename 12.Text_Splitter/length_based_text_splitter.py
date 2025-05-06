from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Load a PDF file
loader = PyPDFLoader("/Users/yogeshagrawal/Desktop/Gen AI/12.Text_Splitter/sample_generative_ai.pdf")

docs = loader.load()

# print(docs)

# text = """
# Space — the final frontier — is a vast, mysterious expanse that stretches far beyond our planet, inspiring human curiosity for centuries. It begins just 100 kilometers above Earth's surface, at the Kármán line, and extends outward into an unimaginably large universe filled with stars, planets, galaxies, black holes, and countless unknowns. Despite centuries of observation and decades of exploration, space remains one of the most fascinating and least understood realms known to humanity.

# At its core, space is largely a vacuum, meaning it contains very little matter. Unlike Earth, it lacks an atmosphere, weather, and air. This vacuum allows light to travel great distances unobstructed, which is why we can see stars that are billions of light-years away. However, it also means that sound cannot travel, and any object placed in space must carry its own life-support systems to survive.

# One of the most significant milestones in our relationship with space came in 1957, when the Soviet Union launched Sputnik 1, the first artificial satellite. This event triggered the space race, leading to monumental achievements such as Yuri Gagarin becoming the first human in space in 1961, and the Apollo 11 moon landing in 1969, when Neil Armstrong took his famous first step on the lunar surface.
# """

char_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

# chunks = char_splitter.split_text(text)
chunks = char_splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)
print(chunks[0].metadata)