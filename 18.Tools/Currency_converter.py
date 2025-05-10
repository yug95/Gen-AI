from langchain_core.tools import tool, InjectedToolArg
from langchain_openai.chat_models import ChatOpenAI
from typing import Annotated
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import requests

load_dotenv()



@tool
def get_conversion_factor(base_currency : str, target_currency:str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert_currency(base_curr_val: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Given a currency conversion ratio multiplies it with base currency value to generate converted value.
    """
    return base_curr_val * conversion_rate


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])


messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]


# print(message_list)
AI_message = llm_with_tools.invoke(messages)
messages.append(AI_message)

# print(AI_message.tool_calls)

for tool_call in AI_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_msg1 = get_conversion_factor.invoke(tool_call)
        # fetch this conversion rate
        # print(tool_msg1)
        conversion_rate = json.loads(tool_msg1.content)['conversion_rate']
        # print(conversion_rate)
        messages.append(tool_msg1)

    if tool_call['name'] == 'convert_currency':
        tool_call['args']['conversion_rate'] = conversion_rate
        # print(tool_call)
        tool_msg2 = convert_currency.invoke(tool_call)
        messages.append(tool_msg2)

# print(messages[3])

result = llm_with_tools.invoke(messages)

print(result)


