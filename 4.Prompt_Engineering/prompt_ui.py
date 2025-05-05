from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

model = ChatOpenAI()

st.header('Research Tool')

# user_input = st.text_input("Enter your Prompt here:")

paper_input = st.selectbox("select the paper", ["Attention is all you need", "Bert pre-training of deep bidirectional transformer", "GPT3 : Language models are few shot learners"])

style_input = st.selectbox("select explanation style", ["Beginner", "Mathematical", "Technical", "Code-oriented"])

lentgh_input = st.selectbox("select explanation length", ["short", "medium", "long"])

# #template
# template = PromptTemplate(
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}
# Explanation Length: {length_input}

# 1. Mathematical Details:

# Include relevant mathematical equations if present in the paper.

# Explain the mathematical concepts using simple, intuitive code snippets where applicable.

# 2. Analogies:

# Use relatable analogies to simplify complex ideas.
# If certain information is not available in the paper, respond with:
# "Insufficient information available" instead of guessing.
# Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
# input_variables=["paper_input", "style_input", "length_input"],
# validate_template=True
# )

#Alternative way to load the template   

template = load_prompt("prompt_template.json")

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": lentgh_input
})


if st.button('summarize'):
    result = model.invoke(prompt)
    st.write(result.content)