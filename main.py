import os
from transformers import AutoModelForCausalLM, AutoTokenizer

save_directory = "./saved_model"

# Check if the model has been saved locally
if os.path.exists(save_directory):
    print("Loading model from local directory...")
    model = AutoModelForCausalLM.from_pretrained(save_directory, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
else:
    print("Downloading and saving model...")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    
    # Save model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

generated_ids = model.generate(encodeds, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
