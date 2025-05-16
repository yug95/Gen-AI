from langchain.prompts import PromptTemplate, ChatPromptTemplate,MessagesPlaceholder

def get_prompt(input_dict):
    system_prompt = (
        """
        You are a helpful medical assistant for question answering task.use the following pieces of context to answer the answer.If you don't know the answer, just say "I don't know".
        \n\n
        {context}
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{chat_history}"),
            ("human", "{input}")
        ]
    )

    return prompt