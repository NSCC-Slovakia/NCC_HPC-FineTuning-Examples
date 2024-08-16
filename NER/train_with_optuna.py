from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import os
import utils
import optuna

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
print(f"Example tokens: {example['tokens']}")
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

# model initialization for hyperparameter search using the built in trainer method
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(id2label.keys()), 
        id2label=id2label, 
        label2id=label2id
    )

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
    save_strategy="no", # no checkpoints will be saved during the training process
    push_to_hub=False,
    remove_unused_columns=False
)
# Initialize your Trainer
trainer = Trainer(
    model=None,
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_dataset["train"].remove_columns(["id", "tokens", "ner_tags"]),
    eval_dataset=tokenized_dataset["validation"].remove_columns(["id", "tokens", "ner_tags"]),
    compute_metrics=utils.compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# very basic grid search just as a demonstration
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [16, 32, 64])
    }

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5
)

best_trial

# Save the final model
#trainer.save_model("models/res1/final_model")

