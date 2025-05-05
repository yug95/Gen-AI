from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# OpenAI can be used
model = ChatOpenAI()

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me the name, age and city of a fictional character \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# without chain
# prompt = template1.invoke({})
# print(prompt)
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

# with chain .....
chain = template1 | model | parser
result = chain.invoke({})
print(result)