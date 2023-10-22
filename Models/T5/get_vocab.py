from transformers import T5Tokenizer
import json

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Get the vocabulary list
vocab_list = tokenizer.get_vocab()

# Create a list of (token, id) tuples
word2id = {token: tokenizer.convert_tokens_to_ids(token) for token in vocab_list}
id2word = {tokenizer.convert_tokens_to_ids(token): token for token in vocab_list}

# Save the vocab list with IDs to a file
with open("word2ids.json", "w", encoding="utf-8") as outfile:
    json.dump(word2id, outfile, indent=4)

with open("ids2word.json", "w", encoding="utf-8") as outfile:
    json.dump(id2word, outfile, indent=4)