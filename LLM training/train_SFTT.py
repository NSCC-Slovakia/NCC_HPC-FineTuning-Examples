from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import AutoModelForSeq2SeqLM

# loading and splitting dataset
df = pd.read_csv('LLM_prompt_train_nostrict_v0.csv')
df = pd.read_csv('../prompt_data/LLM_prompt_train_nostrict_v0.csv')
df.head()

df_dict = df.to_dict(orient='records')
dataset = Dataset.from_dict({'instruction': [item['prompt'] for item in df_dict], 'output': [item['completion'] for item in df_dict]})
dataset = dataset.train_test_split(0.3)

# loading model
model = AutoModelForSeq2SeqLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=True,
    device_map="auto",)

# training
trainer = SFTTrainer(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="instruction",
    max_seq_length=8,
)
trainer.train()

model.save_pretrained("llama3_trained", from_pt=True)

