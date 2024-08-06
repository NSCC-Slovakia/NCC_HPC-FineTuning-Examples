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

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
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
    df.at[index, 'prompt'] = row['question'] + '\nchoose only one of the following options: \n1. ' + row['opa'] + '\n2. ' + row['opb'] + '\n3. ' + row['opc'] + '\n4. ' + row['opd'] + '\nRespond only with the number of the chosen option.'
    
df = df.sample(n=500, random_state=seed) # sample only 500 questions for testing

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

b = 0
for index, row in df.iterrows():
    chat = [
        {"role": "user", "content": row['prompt'] }, 
    ]
    prompt_inputs = tokenizer.apply_chat_template(
        chat, 
        add_generation_prompt=True, 
        return_tensors="pt"
        ).to(model.device)
    attention_mask = torch.ones_like(prompt_inputs)  
    
    prompt_outputs = model.generate(
        prompt_inputs,
        attention_mask=attention_mask,
        max_new_tokens=7,
        eos_token_id=terminators,
        )
    response = prompt_outputs[0][prompt_inputs.shape[-1]:]
    df.loc[index, 'answer'] = tokenizer.decode(response, skip_special_tokens=True)
    print(df[index, 'answer'])
    print(b)
    b = b + 1

print(df.head())

filename = "data/test_mistral"
df.to_csv(filename+".csv", index=False)
