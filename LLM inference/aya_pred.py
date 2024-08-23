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
model = AutoModelForSeq2SeqLM.from_pretrained(
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
    df.at[index, 'prompt'] = row['question'] + '\nchoose only one of the following options: \n0. ' + row['opa'] + '\n1. ' + row['opb'] + '\n2. ' + row['opc'] + '\n3. ' + row['opd'] + '\nRespond only with the number of the chosen option.'
    
df = df.sample(n=500, random_state=seed) # sample only 500 questions for testing, just for the example purposes

# generate responses
b = 0
for index, row in df.iterrows():
    prompt_inputs = tokenizer.encode(row['prompt'], return_tensors="pt").to(model.device)
    prompt_outputs = model.generate(prompt_inputs, max_new_tokens=7)
    df.loc[index, 'answer'] = tokenizer.decode(prompt_outputs[0], skip_special_tokens=True)
    print("Response of the LLM: {}".format(tokenizer.decode(prompt_outputs[0], skip_special_tokens=True)))
    print("{}. sample".format(b))
    b = b + 1

# change the responses to the predictions of correct answer (0, 1, 2, 3)
df['prediction'] = None

for index, row in df.iterrows():
    pom = row["answer"]
    if "0" in pom:
        df.at[index, "prediction"] = 0
    elif "1" in pom:  
        df.at[index, "prediction"] = 1
    elif "2" in pom:
        df.at[index, "prediction"] = 2
    elif "3" in pom:  
        df.at[index, "prediction"] = 3
    else:
        df.at[index, "prediction"] = -1

# calculate accuracy, false positive and invalid predictions
accuracy = sum(df["cop"] == df["prediction"])/df.shape[0]
false_positive = sum(df["cop"]) != df["prediction"]/df.shape[0]
no_invalid = sum(df["prediction"] == -1)/df.shape[0]

print("Accuracy: ", accuracy)
print("False positive: ", false_positive)
print("Invalid: ", no_invalid)

# save the results
filename = "data/test_aya"
df.to_csv(filename+".csv", index=False)

