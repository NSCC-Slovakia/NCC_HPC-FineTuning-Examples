# Load all the neccessary lybraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch



# Enter your access token from HuggingFace (HF)
access_token = " "

# Choose the model, any AutoModelForCausalLM from HF; in case other models use the loading as in the documentation for that model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    token=access_token
)
 
#load Data
prompts = pd.read_csv('prompt_data/LLM_prompt_test_v0.csv')
#prompts = pd.read_csv('LLM_prompt_test_4eprompts.csv')
#prompts = pd.read_csv('LLM_prompt_test_7bprompts.csv')


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
#%%
#cyklus II. moznost
#bude fungovat pre MS prompty v0
b = 0
for index, row in prompts.iterrows():
    # chat = [
    #     {"role": "system", "content": "Choose one of the four intents that best describes the question. If you are unsure, answer 'invalid'. Answer only with the intent name:"},
    #     {"role": "user", "content": row['prompt'][:-150] },
    # ] #system pre MS prompty
    # chat = [
    #     {"role": "system", "content": "Choose the number that corresponds to the question. If you are uncertain, return option 4 (invalid). Your response should only contain the chosen option."},
    #     {"role": "user", "content": row['prompt'][:-165] }, #pre 4e prompty
    # ]
    # chat = [
    #     {"role": "system", "content": "Select the best intent option based on the question. If you are uncertain, please choose 'invalid'. Your response should only contain the name of the chosen option."},
    #     {"role": "user", "content": row['prompt'][:-165] }, #pre 7b prompty
    # ]
    chat = [
        {"role": "system", "content": "You will be given a question/sentence in Slovak language asked by a customer. You are supposed to select the best intent option based on the question. If you are uncertain, or none of the intents describes the question, choose 'invalid'. Your response should only contain the name of the chosen option."},
        {"role": "user", "content": row['prompt'][:-150] }, #novy system content + MS prompt
    ]
    # chat = [
    #     {"role": "system", "content": row['prompt']} 
    # ]
    # chat = [
    #     {"role": "user", "content": row['prompt']},
    # ]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)  
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    #print(answer)
    prompts.at[index, "answer"] = answer
    # print(b)
    # b = b + 1
    # if b >= 10:
    #     break
#%%
print(prompts.head())

prompts.to_csv("LLAMA3_8bi_LLM_prompt_test_MS_changedchat.csv", index=False)

