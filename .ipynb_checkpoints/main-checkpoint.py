from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token='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')
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token='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')

prompt = """
You are an AI assistant and you are very good at doing sentiment classification.

You are only allowed to choose one of the following 3 categories: Negative,
Neurtal, or Positive. Please provide only one category for each product in
JSON format where the key is the index for each product and the
value is one of the 3 categories. For example: {1: Negative}. Do not repeat
or return the content back again, just provide the category in the defined format.

Classify the following paragraph: 
The movie was neither good or bad, it was ok.
"""

messages = [
    {"role": "user", "content": prompt}
]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
