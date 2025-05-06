from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space — the final frontier — is a vast, mysterious expanse that stretches far beyond our planet, inspiring human curiosity for centuries. It begins just 100 kilometers above Earth's surface, at the Kármán line, and extends outward into an unimaginably large universe filled with stars, planets, galaxies, black holes, and countless unknowns. Despite centuries of observation and decades of exploration, space remains one of the most fascinating and least understood realms known to humanity.

At its core, space is largely a vacuum, meaning it contains very little matter. Unlike Earth, it lacks an atmosphere, weather, and air. This vacuum allows light to travel great distances unobstructed, which is why we can see stars that are billions of light-years away. However, it also means that sound cannot travel, and any object placed in space must carry its own life-support systems to survive.
""" 

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)

# Split the text into chunks
chunks = text_splitter.split_text(text)

# Print the number of chunks and the first chunk
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0]}")