#!/usr/bin/env python3
"""Chat with the quantized model."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "quantized_models/chat_model"

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
print("Ready. Type 'quit' to exit.\n")

while True:
    try:
        user = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user:
        continue
    if user.lower() == "quit":
        break
    inputs = tokenizer(user, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=True,
            temperature=0.7, top_p=0.9,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{response}\n")
