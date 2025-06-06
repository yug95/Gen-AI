from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import datetime
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage 

from dotenv import load_dotenv
load_dotenv()

actor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a expert AI researcher.
            current_time: {time}

        1. {first_instruction}
        2. reflect and critique your answer. Be severe to maximize improvement.
        3. After the reflection, ** list 1-3 search queries separately** for researching improvments.
        Do not include them inside the reflection.
        """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(time = lambda : datetime.datetime.now().isoformat())


first_prompt = actor_prompt.partial(
    first_instruction="Provide a detailed ~250 words answer."
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revise_prompt = actor_prompt.partial(
    first_instruction=revise_instructions
) 

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000)

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

first_response_chain = first_prompt | llm.bind_tools(tools = [AnswerQuestion], tool_choice="AnswerQuestion")
revise_response_chain = revise_prompt | llm.bind_tools(tools = [ReviseAnswer], tool_choice="ReviseAnswer")


# response = first_response_chain.invoke({
#     "messages":[HumanMessage(content="Write me a blog post about how AI can help small businesses.")],
# })

# print(response)
