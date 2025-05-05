from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# Create a WebBaseLoader instance
loader = WebBaseLoader("https://www.geeksforgeeks.org/what-is-web-scraping/")
# Load the documents
docs = loader.load()

print(docs[0].page_content)

model = ChatOpenAI()
output_parser = StrOutputParser()
prompt = PromptTemplate(
    template="Answer the {question} from following {text}",
    input_variables=["question","text"]
)

chain = prompt | model | output_parser

result = chain.invoke({
    'question': "What is web scraping?",
    'text': docs[0].page_content
})

print(result)  # This will print the random response from the LLM