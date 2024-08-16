import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import DatasetDict, Dataset
import numpy as np
import torch
import time
import argparse
from seqeval.metrics import accuracy_score
from sklearn.metrics import classification_report

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
args = parser.parse_args()

# Check available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print("Number of available GPUs:", num_gpus)

# Load and clean data
df = pd.read_csv('Twitter_Data.csv')
df = df.dropna(subset=['category', 'clean_text'])  # Drop rows with NaN values

# Mapping categories to integer labels
mapping = {-1: 0, 0: 1, 1: 2}
df['category_1'] = df['category'].replace(mapping).astype(int)

# Print number of samples per category
rows_per_ctg = df['category_1'].value_counts()
print("Number of samples per category:")
print(rows_per_ctg)

# Split data into train, validation, test sets
train_data, test_data = train_test_split(df.to_dict(orient='records'), test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Create DatasetDict for train, validation, test datasets
dataset_dict = DatasetDict({
    'train': Dataset.from_dict({'text': [item['clean_text'] for item in train_data], 'label': [item['category_1'] for item in train_data]}),
    'validation': Dataset.from_dict({'text': [item['clean_text'] for item in val_data], 'label': [item['category_1'] for item in val_data]}),
    'test': Dataset.from_dict({'text': [item['clean_text'] for item in test_data], 'label': [item['category_1'] for item in test_data]})
})

# Initialize BERT tokenizer
MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=True)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Tokenize datasets and format for PyTorch
tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
for split in ['train', 'validation', 'test']:
    tokenized_dataset[split] = tokenized_dataset[split].map(lambda example: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in example.items()})

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(predictions, labels)}

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=3)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    output_dir="bert_twitter",
    num_train_epochs=4,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    eval_steps=300,
    save_strategy="epoch",
    save_steps=300,
    load_best_model_at_end=True,
    eval_accumulation_steps=300,
    logging_steps=300
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training and evaluation
start_time = time.time()
trainer.train()
end_time = time.time()
duration = end_time - start_time
print("Training duration:", duration, "seconds")

# Save model
model.save_pretrained("bert_sentiment_model")

# Loading the model 
# model = BertForSequenceClassification.from_pretrained("bert_sentiment_model")

# Evaluate the model on the test dataset
results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])

# Print evaluation results
print(results)

# Get precision, recall, f1-score and support for the test dataset
predictions = trainer.predict(tokenized_dataset['test'])
predicted_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = tokenized_dataset['test']['label']

# Compute classification report
class_report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2'])

print("Classification Report:")
print(class_report)