from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments, pipeline, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from huggingface_hub import login, notebook_login
import os
from utils import prompt_instruction_format, LossThresholdCallback


data_files = {'train':'../data/train.csv','test':'../data/test.csv'}

dataset = load_dataset('csv',data_files=data_files)

# model_name = "lucadiliello/flan-t5-base"
# google/flan-t5-base
# google/flan-t5-small

model_name = "lucadiliello/bart-small"


save_model_name = model_name.split("/")[1]

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# save model to model_name/model directory
# model.save_pretrained(f'{save_model_name}/model')

# Load the model from the model_name/model directory
model = AutoModelForSeq2SeqLM.from_pretrained(f'../{save_model_name}/model')
print("Loaded model!")

#Makes training faster but a little less accurate 
model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# save tokenizer to model_name/tokenizer directory
# tokenizer.save_pretrained(f'{save_model_name}/tokenizer')

# Load the tokenizer from the model_name/tokenizer directory
tokenizer = AutoTokenizer.from_pretrained(f'../{save_model_name}/tokenizer')
print("Loaded tokenizer!")

#setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Set up TrainingArguments
trainingArgs = TrainingArguments(
    output_dir="output",
    num_train_epochs=10,
    per_device_train_batch_size=4 if "large" not in model_name else 2,
    evaluation_strategy="steps",  # Run evaluation every few steps
    eval_steps=100,               # Validate every 500 steps
    save_strategy="epoch",        # Save checkpoint at the end of each epoch
    learning_rate=2e-4,
)

peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM",
)

# Set the loss threshold
loss_threshold = 0.1  # Example threshold
# Initialize the custom callback with the output directory
# Initialize the custom callback with the model and tokenizer
loss_threshold_callback = LossThresholdCallback(
    threshold=loss_threshold,
    output_dir=trainingArgs.output_dir,
    model=model,
    tokenizer=tokenizer
)

# Set up the trainer
# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=trainingArgs,
    callbacks=[loss_threshold_callback]  # Add custom callback here
)

trainer.train()