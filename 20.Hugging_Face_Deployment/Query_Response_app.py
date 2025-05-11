from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import streamlit as st


st.title("Query Response APP")

load_dotenv()

llm = ChatOpenAI()

prompt = PromptTemplate(
    template= 'Please write your query : {Query}',
    input_variables=['Query']
)

parser = StrOutputParser()

chain = prompt | llm | parser

Query = st.text_input("Query: ", key="Query")

submit = st.button("Ask the question")

# Query = 'What is the capital of India'

if submit:

    response = chain.invoke({'Query':Query})

    st.subheader('the response is')
    st.write(response)

    print(response)