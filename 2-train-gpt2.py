import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("csv", data_files="preprocessed_amazon_reviews.csv")

dataset = dataset['train'].train_test_split(test_size=0.2) # Split into train/test

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Adjust the model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.to(device)


metric = evaluate.load('accuracy')


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8, # Automatically adapts to the GPU memory if available
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=5e-5,
    weight_decay=0.01,
    save_steps=100,
    save_total_limit=2, # Limit saved checkpoints to save disk space
    load_best_model_at_end=True, # Load the best model at the end of training
    fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=metric,
)