from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]") # First 5000 stories

with open("dataset.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["text"] + "\n")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# This is added to avoid spaces between tokens from the same word
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=1000, 
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[END]"]
)

files = ["dataset.txt"]
tokenizer.train(files, trainer)

tokenizer.save("mini-gpt-tokenizer.json")