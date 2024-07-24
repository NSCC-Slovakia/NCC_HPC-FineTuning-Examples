from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101")

# load only the test data
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["test"])
df = df[df['choice_type'] != 'multi'] # only single choice questions, multi choice questions are not properly defined by the dataset

# create prompts for the model
df['prompt'] = None
df['answer'] = None
for index, row in df.iterrows():
    df.at[index, 'prompt'] = row['question'] + '\nchoose only one of the following options: \na. ' + row['opa'] + '\nb. ' + row['opb'] + '\nc. ' + row['opc'] + '\nd. ' + row['opd']


b = 0
for index, row in df.iterrows():
    #print(row['prompt'])
    prompt_inputs = tokenizer.encode(row['prompt'], return_tensors="pt")
    prompt_outputs = model.generate(prompt_inputs, max_new_tokens=7, temperature = 0)
    df['answer'][index] = tokenizer.decode(prompt_outputs[0])
    print(tokenizer.decode(prompt_outputs[0]))
    b = b + 1
    if b >= 10:
        break

print(df.head())

filename = "data/test_aya"
df.to_csv(filename+".csv", index=False)
