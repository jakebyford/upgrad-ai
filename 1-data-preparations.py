import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("amazon_polarity")

print(dataset['train'][0])

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    return text

dataset = dataset.map(lambda x: {'clean_text': clean_text(x['content'])})

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['clean_text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format(type='torch', device=device)

train_data = pd.DataFrame(tokenized_dataset['train'])

train_data.to_csv('preprocessed_amazon_reviews.csv', index=False)

print("Preporcessed data saved to preprocessed_amazon_reviews.csv")
