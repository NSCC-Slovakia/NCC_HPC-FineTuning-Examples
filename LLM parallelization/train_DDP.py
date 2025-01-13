import torch
import re
import random
import pandas as pd
from trl import SFTTrainer
import pynvml
from peft import LoraConfig
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM
from accelerate import PartialState


def print_gpu_utilization():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_used = []
    for device_index in range(device_count):
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        memory_used.append(device_info.used/1024**3)
    print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])
df = df[df['choice_type'] != 'multi'] # only single choice questions, multi choice questions are not properly defined by the dataset

# create prompts for the model
df['prompt'] = None
for index, row in df.iterrows():
    df.at[index, 'prompt'] = row['question'] + '\nchoose only one of the following options: \n0. ' + row['opa'] + '\n1. ' + row['opb'] + '\n2. ' + row['opc'] + '\n3. ' + row['opd'] + '\nRespond only with the number of the chosen option.'
    
df = df.sample(n=20000, random_state=42) # sample only 20000 questions for training, just for the example purposes

# create training dataset
df_dict = df.to_dict(orient='records')

dataset = Dataset.from_dict({'instruction': [item['prompt'] for item in df_dict], 'output': [str(item['cop'])+"." for item in df_dict]})
dataset = dataset.train_test_split(0.3)


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048

# Set padding strategy because of warning  
tokenizer.padding_side = 'right'

# define chat template, must be same in the prediction as in the training!
def apply_chat_template(example, tokenizer):
    chat = [
        {"role": "user", "content": example['instruction']},
        { "role": "assistant", "content": example['output'] },
    ]
    example["text"] = tokenizer.apply_chat_template(chat, tokenize=False)
    return example

column_names = list(dataset["train"].features)

raw_datasets = dataset.map(apply_chat_template,
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)

# create the train test splits for the training
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

for index in random.sample(range(len(raw_datasets["train"])), 3):
  print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
  
# initialize the PartialState (for the accekerate)
ps = PartialState()
num_processes = ps.num_processes
process_index = ps.process_index
local_process_index = ps.local_process_index

# specify how to quantize the model, this save a lot of memory and speed up the training
# if you want to quantize it uncomment the line in the model_kwargs
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
  model_id,
    attn_implementation="sdpa", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map={'':local_process_index},
    # quantization_config=quantization_config, # uncomment this line to enable quantization
)

# path where the Trainer will save its checkpoints and logs
output_dir = 'data/mistral_trained'

# hyperparameters for the training
training_args = TrainingArguments(
    bf16=True, 
    do_eval=True,
    eval_strategy="epoch",
    per_device_eval_batch_size=1, 
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-04,
    log_level="info",
    logging_steps=5,
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=3,
    output_dir=output_dir,
    overwrite_output_dir=True,
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard", 
    # figure out how to save best model
    save_strategy="epoch",
    save_total_limit=None,
    seed=42,
)

# Peft configuration, enables to train only the adapters to the model, not whole model
peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
print(train_dataset)

# create the trainer
trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

if process_index == 0:  # Only print in first process.
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
        
# train the model
train_result = trainer.train()

# Print statistics in first process only:
if process_index == 0:
    print(f"Run time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"{num_processes} GPUs used.")
    print(f"Training speed: {train_result.metrics['train_samples_per_second']:.1f} samples/s (={train_result.metrics['train_samples_per_second'] / num_processes:.1f} samples/s/GPU)")

# Print memory usage once per node:
if local_process_index == 0:
    print_gpu_utilization()
    
# save the results
metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

if process_index == 0:
    trainer.save_model()

