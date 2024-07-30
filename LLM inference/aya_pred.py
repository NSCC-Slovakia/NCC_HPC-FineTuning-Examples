from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_id = "CohereForAI/aya-101"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretraine(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map='cuda'
    )

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
    prompt_outputs = model.generate(prompt_inputs, max_new_tokens=7)
    df.loc[index, 'answer'] = tokenizer.decode(prompt_outputs[0], skip_special_tokens=True)
    print(b)
    b = b + 1

print(df.head())

filename = "data/test_aya"
df.to_csv(filename+".csv", index=False)
