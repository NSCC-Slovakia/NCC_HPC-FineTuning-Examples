import time
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, AutoTokenizer
from tqdm import tqdm
import medmcqa_helper
import gpu_helper


model_id = 'mistral7b-medmcqa-1'

model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',  # 'eager', 'sdpa', or "flash_attention_2"
    device_map='cuda'
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

generation_config = GenerationConfig(
    do_sample=True,
    top_k=20,
    top_p=0.8,
    temperature=0.1,
    max_new_tokens=64,
    pad_token_id=tokenizer.pad_token_id
)

data_val = load_dataset('medmcqa', split='validation')
data_val = data_val.map(lambda entry:medmcqa_helper.add_prompt(entry, tokenizer, include_answer=False), load_from_cache_file=False)

def solve_questions(question_prompts):
    # Convert prompts to tokens:
    inputs = tokenizer(question_prompts, padding=True, return_tensors="pt").to("cuda")
    # Get model predictions:
    outputs = model.generate(**inputs, generation_config=generation_config)
    # Remove input prompts from beginning of outputs:
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    # Convert tokens to prompts:
    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answers

start_time = time.time()
batch_size = 64
correct_count = 0
for i in tqdm(range(0, len(data_val), batch_size)):
    entries = data_val.select(range(i, min(i+batch_size, len(data_val))))
    question_prompts = entries['text']
    predictions = solve_questions(question_prompts)
    for entry, prediction in zip(entries, predictions):
        if prediction.startswith('Answer: '):
            prediction = prediction[8:]
        if medmcqa_helper.check_answer(entry, prediction):
            correct_count += 1
end_time = time.time()

correct_percentage = 100 * correct_count / len(data_val)

print(f"{correct_percentage:.2f}% ({correct_count} out of {len(data_val)}) answers correct.")
print(f"Run time: {end_time - start_time:.2f} seconds")
print(f"Samples/second: {len(data_val) / (end_time - start_time):.1f}")
gpu_helper.print_gpu_utilization()
