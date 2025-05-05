from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("who is pm of india")
print(result.content)


# from langchain.llms import HuggingFaceHub
# from langchain import PromptTemplate, LLMChain
# import os

# # Set your Hugging Face token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# # Set up the LLM from Hugging Face
# llm = HuggingFaceHub(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     model_kwargs={"temperature": 0.7, "max_new_tokens": 100}
# )

# # Create a prompt template
# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="You are a helpful assistant. Answer the following question:\n\nQuestion: {question}\nAnswer:"
# )

# # Set up LangChain pipeline
# chain = LLMChain(llm=llm, prompt=prompt)

# # Ask a question
# response = chain.invoke("What is the capital of France?")
# print(response)