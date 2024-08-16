import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def main(args):
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    # Create NER pipeline
    ner = pipeline("ner", model=model, tokenizer=tokenizer)

    # Example usage
    text = "Apple is expected to launch new products next week in California."
    predictions = ner(text)
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved NER model")
    args = parser.parse_args()
    
    main(args)