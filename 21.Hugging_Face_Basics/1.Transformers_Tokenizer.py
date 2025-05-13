from transformers import AutoTokenizer, AutoModelForCausalLM

text = "I was not happy with ther service."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# print(tokenizer.tokenize(text))

input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
# print(input_ids)

# encoded_input = tokenizer(text)
# print(encoded_input)

decoded_input = tokenizer.decode(input_ids)
print(decoded_input)