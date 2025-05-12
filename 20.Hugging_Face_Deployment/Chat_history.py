from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.header("Chatting App!")

llm = ChatOpenAI

chat_list = [SystemMessage("You are a helpful Assistant")]

submit = st.button("Submit")
user = st.text_input(label="user", key="user")
print(user)
while True:
    
    if user == 'exit':
        break
    
    if submit:
        result = llm.invoke(user)
        chat_list.append(HumanMessage(content=user))
        chat_list.append(AIMessage(content=result.content))
        st.subheader('Your response is')
        st.write(result.content)
        st.subheader('Your history')
        st.write(chat_list)



