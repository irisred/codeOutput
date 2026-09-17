import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "../Meta-Llama-3-8B"
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
print("moving to cuda...")
device = torch.device("cuda:0")
model = model.to(device)
print("done")
