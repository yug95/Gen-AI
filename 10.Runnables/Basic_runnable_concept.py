import random

class Duplicate_LLM:
    def __init__(self, name):
        print("LLM Created")
    
    def predict(self, prompt):

        response_list = ["Hello, how can I assist you today?", "What can I do for you?", "How may I help you?"]
        return {'response':random.choice(response_list)}
    

llm = Duplicate_LLM("MyLLM")
# print(llm.predict("Hello"))  # This will print a random response from the list
    

class Duplicate_Prompt_Template:
    def __init__(self, template, Input_variables):
        self.template = template
        self.input_variables = Input_variables
        print("Prompt Template Created")
    
    def format(self, input_dict):
        return self.template.format(**input_dict)
    

prompt = Duplicate_Prompt_Template(
    template=" Write a poem {length} about {topic}",
    Input_variables=["length","topic"]
)

prompt1 = prompt.format({"length":5,"topic":"Python"})  # This will print the formatted prompt with the topic "Python"
# result = llm.predict(prompt1)  # This will print a random response from the list
# print(result)  # This will print the random response from the LLM



class Duplicate_chain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
        print("Chain Created")
    
    def run(self, input_dict):
        formatted_prompt = self.prompt.format(input_dict)
        return self.llm.predict(formatted_prompt)['response']
    
chain = Duplicate_chain(llm, prompt1)
result = chain.run({"length":5,"topic":"Python"})  # This will print a random response from the list
print(result)  # This will print the random response from the LLM