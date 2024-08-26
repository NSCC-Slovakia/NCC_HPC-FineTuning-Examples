from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import os
import utils
import pandas as pd
import matplotlib.pyplot as plt

# Load the CoNLL 2003 dataset, which includes named entities annotated in BIO format (Inside, Outside, Beginning)
dataset = load_dataset("conll2003", trust_remote_code = True)

# Remove unnecessary columns (pos_tags, chunk_tags) from the dataset
dataset = dataset.remove_columns(['pos_tags', 'chunk_tags'])

# Extract the names of the NER labels from the dataset
label_list = dataset["train"].features["ner_tags"].feature.names

# Define the model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True)

# Tokenize the entire dataset (all splits)
tokenized_dataset = dataset.map(
    lambda examples: utils.tokenize_and_align_labels(examples, tokenizer=tokenizer), batched=True)

tokenized_dataset

# Display an example from the tokenized dataset
example = tokenized_dataset['train'][0]
print(f"Example keys: {example.keys()}")
print(f"Input IDs: {example['input_ids']}")
print(f"Labels: {example['labels']}")
print(f"Attention Mask: {example['attention_mask']}")

# in charge of batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)    

# Define label mappings for the model
label2id = {label: index for index, label in enumerate(label_list)}
id2label = {value: key for key, value in label2id.items()}
num_labels = len(label_list)

print(f"Label to ID mapping: {label2id}")
print(f"ID to Label mapping: {id2label}")

# Initialize the model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=num_labels, 
    id2label=id2label, 
    label2id=label2id).cuda()

# Define the training arguments
training_args = TrainingArguments(
    output_dir="../models/res1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy='epoch',
    metric_for_best_model="eval_loss",
    save_strategy="epoch", 
    save_total_limit=5,
    push_to_hub=False,
    remove_unused_columns=False,
    load_best_model_at_end=True
)
# Initialize your Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].remove_columns(["id", "tokens", "ner_tags"]),
    eval_dataset=tokenized_dataset["validation"].remove_columns(["id", "tokens", "ner_tags"]),
    compute_metrics=utils.compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("models/final_model")

# check the results on the test dataset
test_metrics = trainer.predict(tokenized_dataset['test'].remove_columns(["id", "tokens", "ner_tags"])).metrics
print(f"Test set metrics:\n{test_metrics}")


train_logs = []
valid_logs = []
for index, x in enumerate(trainer.state.log_history):
    if 'loss' in x.keys():
        train_logs.append(x)
    elif 'eval_loss' in x.keys():
        valid_logs.append(x)
    else:
        pass

train_logs = pd.DataFrame(train_logs)
valid_logs = pd.DataFrame(valid_logs)
logs = train_logs.merge(valid_logs, on=["epoch", "step"])

# Visualize and save the graphs
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(logs['epoch'], logs['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(logs['epoch'], logs['eval_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap of labels
plt.tight_layout()

# Save the figure as a PNG image
plt.savefig('training_validation_loss.png')

# Display the plot if needed (optional)
plt.show()
