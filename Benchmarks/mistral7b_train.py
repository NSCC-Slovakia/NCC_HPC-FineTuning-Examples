import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
# import wandb
import medmcqa_helper
import gpu_helper


model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
max_seq_length = 1024

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = 'right'

data = load_dataset('medmcqa', split='train')
data = data.map(lambda entry:medmcqa_helper.add_prompt(entry, tokenizer, include_answer=True), load_from_cache_file=True)

ps = PartialState()
num_processes = ps.num_processes
process_index = ps.process_index
local_process_index = ps.local_process_index

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    attn_implementation='sdpa',  # 'eager', 'sdpa', or "flash_attention_2"
    device_map={'':local_process_index}
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1  # disable tensor parallelism

peft_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=16,
    lora_alpha=32,  # rule: lora_alpha should be 2*r
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
)

project_name = 'mistral7b-medmcqa'
run_name = '1'
# notes = ''

# wandb_run = wandb.init(
#     project=project_name,
#     name=run_name,
#     notes=notes
# )

training_arguments = TrainingArguments(
    # When using newer versions of `trl`, use SFTConfig(...) instead of TrainingArguments(...).
    output_dir=f'{project_name}-{run_name}',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True, # Gradient checkpointing improves memory efficiency, but slows down training,
        # e.g. Mistral 7B with PEFT using bitsandbytes:
        # - enabled: 11 GB GPU RAM and 12 samples/second
        # - disabled: 40 GB GPU RAM and 8 samples/second
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Use newer implementation that will become the default.
    ddp_find_unused_parameters=False,  # Set to False when using gradient checkpointing to suppress warning message.
    log_level_replica='error',  # Disable warnings in all but the first process.
    optim='adamw_torch',  # 'paged_adamw_32bit' can save GPU memory
    learning_rate=2e-4,  # QLoRA suggestions: 2e-4 for 7B or 13B, 1e-4 for 33B or 65B
    warmup_steps=200,
    lr_scheduler_type='cosine',
    logging_strategy='steps',  # 'no', 'epoch' or 'steps'
    logging_steps=50,
    save_strategy='no',  # 'no', 'epoch' or 'steps'
    # save_steps=2000,
    # num_train_epochs=5,
    max_steps=100,
    fp16=True,  # mixed precision training: faster, but uses more memory
    # hub_private_repo=True,
    # report_to='wandb',  # enable wandb
    report_to='none',  # disable wandb
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=data,
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=False,
    # When using newer versions of `trl`, the argument `training_arguments` should be given as
    # an instance of SFTConfig(...) instead of TrainingArguments(...) and the following
    # parameters should be specified there instead of here:
    dataset_text_field='text',
    max_seq_length=max_seq_length,
)

if process_index == 0:  # Only print in first process.
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

result = trainer.train()

# Print statistics in first process only:
if process_index == 0:
    print(f"Run time: {result.metrics['train_runtime']:.2f} seconds")
    print(f"{num_processes} GPUs used.")
    print(f"Training speed: {result.metrics['train_samples_per_second']:.1f} samples/s (={result.metrics['train_samples_per_second'] / num_processes:.1f} samples/s/GPU)")

# Print memory usage once per node:
if local_process_index == 0:
    gpu_helper.print_gpu_utilization()

# Save model in first process only:
if process_index == 0:
    trainer.save_model()

# trainer.push_to_hub()

# wandb.finish()
