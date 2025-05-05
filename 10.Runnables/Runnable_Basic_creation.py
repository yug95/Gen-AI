import random
from abc import ABC, abstractmethod


class Duplicate_Runnable(ABC):
    @abstractmethod
    def invoke(self, input_dict):
        pass


class Duplicate_LLM(Duplicate_Runnable):
    def __init__(self, name):
        print("LLM Created")
    
    def predict(self, prompt):

        response_list = ["Hello, how can I assist you today?", "What can I do for you?", "How may I help you?"]
        return {'response':random.choice(response_list)}
    
    def invoke(self, prompt):
        response_list = ["Hello, how can I assist you today?", "What can I do for you?", "How may I help you?"]
        return {'response':random.choice(response_list)}

llm = Duplicate_LLM("MyLLM")


class Duplicate_Prompt_Template(Duplicate_Runnable):
    def __init__(self, template, Input_variables):
        self.template = template
        self.input_variables = Input_variables
        print("Prompt Template Created")
    
    def format(self, input_dict):
        return self.template.format(**input_dict)
    
    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    

prompt = Duplicate_Prompt_Template(
    template=" Write a poem {length} about {topic}",
    Input_variables=["length","topic"]
)


class Runnble_chain(Duplicate_Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
        print("Chain Created")
    
    def invoke(self, input_dict):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_dict)
        return input_data

chain = Runnble_chain([prompt,llm])
result = chain.invoke({"length":5,"topic":"Python"})  # This will print a random response from the list
print(result)  # This will print the random response from the LLM