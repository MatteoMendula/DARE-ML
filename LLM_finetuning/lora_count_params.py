from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments, pipeline, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from huggingface_hub import login, notebook_login
import os
from utils import prompt_instruction_format, LossThresholdCallback
import numpy as np
import time

data_files = {'train':'../data/train.csv','test':'../data/test.csv'}

dataset = load_dataset('csv',data_files=data_files)

model_name1 = "google/flan-t5-base"
model_name2 = "lucadiliello/bart-small"
model_name3 = "google/flan-t5-small"

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

for model_name in [model_name1, model_name2, model_name3]:

    latency = []
    for _ in range(10):
        start = time.time()
        save_model_name = model_name.split("/")[1]
        model = AutoModelForSeq2SeqLM.from_pretrained(f'../{save_model_name}/model')
        end = time.time()
        latency += [end-start]


    print(f"Model: {model_name}", count_trainable_parameters(model), "Time taken: ", np.mean(latency), np.std(latency))

