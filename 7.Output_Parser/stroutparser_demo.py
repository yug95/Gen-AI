from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
#     # pipeline_kwargs= {"temperature": 0.7, "max_new_tokens": 100}
# )

## Hugging face can be used
# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# # OpenAI can be used
model = ChatOpenAI()



# #1st prompt - > detailed report
template1 = PromptTemplate(
    template="Generate a detailed report of the following: {topic}",
    input_variables=["topic"],
)
# # 2nd prompt - > summary
template2 = PromptTemplate(
    template=" Write the 5 line summary of the following: {text}",
    input_variables=["text"],
)

# Option 1 when we dont want to use parser

# prompt1 = template1.invoke({"topic": "Blackhole"})
# # print(prompt1)
# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result.content})
# # print(prompt2)
# result1 = model.invoke(prompt2)
# print(result1.content)

# Option 2 when we want to use parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Blackhole"})

print(result)
