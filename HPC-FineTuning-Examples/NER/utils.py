import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# do not forget to update the label_list acc. to your data
label_list = [
    "O",       # Outside of a named entity
    "B-PER",   # Beginning of a person's name
    "I-PER",   # Inside of a person's name
    "B-ORG",   # Beginning of an organization name
    "I-ORG",   # Inside of an organization name
    "B-LOC",   # Beginning of a location name
    "I-LOC",   # Inside of a location name
    "B-MISC",  # Beginning of a miscellaneous entity name
    "I-MISC"   # Inside of a miscellaneous entity name
]

# Define a function to compute NER metrics using seqeval
def compute_metrics(p):
    """
    Computes NER metrics: precision, recall, F1, and accuracy.
    Args:
        p (tuple): Tuple containing predictions and true labels.
    Returns:
        dict: Dictionary with computed metrics.
    """    
    predictions, labels = p 
    predictions = np.argmax(predictions, axis=2) # predictions are converted from logits to token
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)
    return {"precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy}

# Tokenize and align labels for token classification
def tokenize_and_align_labels(examples, tokenizer, label_all_word_parts: bool=False):
    """
    Tokenizes examples and aligns NER labels with tokenized inputs.

    Args:
        examples (dict): Dictionary containing "tokens" and "ner_tags".
        tokenizer (PreTrainedTokenizer): Tokenizer object for tokenization.
        label_all_word_parts (bool): Whether to label all word parts for NER.

    Returns:
        dict: Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = [] # Initialize an empty list to store aligned label sequences
    for i, label in enumerate(examples[f"ner_tags"]): # Iterate over each example's NER tags and tokens
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Retrieve the word IDs corresponding to the current example's tokens
        previous_word_idx = None # Initialize variables to keep track of previous word index and label IDs
        label_ids = []
        for word_idx in word_ids:  # Iterate over word IDs to align labels with the tokenized input
            if word_idx is None:
                label_ids.append(-100) # Set the special tokens to -100.
            elif label_all_word_parts or (word_idx != previous_word_idx): # Only label the first token of a given word
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs