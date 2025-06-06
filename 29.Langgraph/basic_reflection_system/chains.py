from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# generation agent task is to generate the best tweet possible based on user request
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a twitter techie influencer assistant tasked with writing excellent twitter post."
            " Generate the best tweet possible based on user request"
            "If user provides critique, respond it with a revised version of your previous attempt. "
         ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# reflection agent task is to reflect on the generated tweet and provide feedback`
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a viral twitter techie influencer gradind a tweet. Generate critique and recommendation for the user's tweet"
            "Always provide detailed recommendation, including request for length, virality, and style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm


