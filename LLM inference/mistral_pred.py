from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# load only the test data
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["validation"])
df = df[df['choice_type'] != 'multi'] # only single choice questions, multi choice questions are not properly defined by the dataset

# create prompts for the model
df['prompt'] = None
df['answer'] = None
for index, row in df.iterrows():
    df.at[index, 'prompt'] = row['question'] + '\nchoose only one of the following options: \na. ' + row['opa'] + '\nb. ' + row['opb'] + '\nc. ' + row['opc'] + '\nd. ' + row['opd']
    
df = df.sample(n=500, random_state=seed) # sample only 500 questions for testing

b = 0
for index, row in df.iterrows():
    #print(row['prompt'])
    prompt_inputs = tokenizer.encode(row['prompt'], return_tensors="pt")
    prompt_inputs['pad_token_id'] = tokenizer.eos_token_id
    prompt_inputs['attention_mask'] = prompt_inputs['input_ids'].ne(tokenizer.pad_token_id).int()
    prompt_outputs = model.generate(prompt_inputs, max_new_tokens=7)
    df.loc[index, 'answer'] = tokenizer.decode(prompt_outputs[0], skip_special_tokens=True)
    print(tokenizer.decode(prompt_outputs[0], skip_special_tokens=True))
    print(b)
    b = b + 1

print(df.head())

filename = "data/test_mistral"
df.to_csv(filename+".csv", index=False)
