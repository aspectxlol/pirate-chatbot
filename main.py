import torch
from transformers import pipeline

<<<<<<< HEAD
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
=======
pipe = pipeline(
    "text-generation",
    model="gpt2",
    device_map={"": "disk"},  # Explicit disk offload
    torch_dtype="auto",
>>>>>>> 0b52b8e (cool fixes)
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Jelaskan tentang berpikir komputasional!"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
