from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import torch
import numpy as np
import random
import pynvml
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def print_gpu_utilization():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_used = []
    for device_index in range(device_count):
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        memory_used.append(device_info.used/1024**3)
    print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')


model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    # torch_dtype=torch.bfloat16, # comment this line when using quantization
    device_map='cuda',
    quantization_config=quantization_config, # uncomment this line to enable quantization
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

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

start_time = time.time()

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
        max_new_tokens=20,
        eos_token_id=terminators,
        )
    response = prompt_outputs[0][prompt_inputs.shape[-1]:]
    df.loc[index, 'answer'] = tokenizer.decode(response, skip_special_tokens=True)
    print("Response of the LLM: {}".format(tokenizer.decode(response, skip_special_tokens=True)))
    print("{}. sample".format(b))
    b = b + 1
end_time = time.time()

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
false_positive = sum(df["cop"] != df["prediction"])/df.shape[0]
no_invalid = sum(df["prediction"] == -1)/df.shape[0]

print("Accuracy: ", accuracy)
print("False positive: ", false_positive)
print("Invalid: ", no_invalid)

# print the GPU utilization and benchmarks
print(f"Run time: {end_time - start_time:.2f} seconds")
print(f"Samples/second: {len(df) / (end_time - start_time):.1f}")
print_gpu_utilization()

# save the results
filename = "data/test_mistral"
df.to_csv(filename+".csv", index=False)
