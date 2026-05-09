import torch
from model.architecture.gpt import GPT, GPTConfig
from model.architecture.tokenizer import CustomTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = GPTConfig(
    vocab_size=50304,
    block_size=256,
    n_layer=12,
    n_head=12,
    n_embd=768
)
model = GPT(model_config)

weights = torch.load("model.pt", map_location=device)["model_state_dict"]
model.load_state_dict(weights)
model.eval()  

tokenizer = CustomTokenizer()
prompt = torch.tensor([tokenizer.encode("""Once upon a time there was a little girl named Sarah. She was only three years old, but she was very brave. One day, Sarah was walking through a forest, when she saw a big, scary bear. Sarah was so scared that she started to cry.

The bear saw Sarah crying and he came closer. He said, "Don't be scared, little girl. I won't hurt you. I'm just looking for something to eat."

Sarah was still scared, but she decided to trust the bear. He said, "I'm looking for something to eat. Would you like to help me?"

Sarah said, "Yes, I'd like to help you." She looked around and found some delicious berries. She said, "Here, eat these. They will make you feel better."

The bear ate the berries and smiled. He said, "Thank you for trusting me. I'm glad I met you."

Sarah smiled back and said, "You""")], device=device)
output = model.generate(prompt, max_new_tokens=100, temperature=0.5, top_k=10)
print(tokenizer.decode(output[0].tolist()))