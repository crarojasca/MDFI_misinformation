# Importing the libraries needed
import pandas as pd
import numpy as np
from transformers import RobertaModel, RobertaTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import (
    HfArgumentParser, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer, 
    AutoModelForSequenceClassification, Trainer
)

import json
from tqdm import tqdm
from pathlib import Path

from dataset import ClaimsData
from arguments import ModelArguments, DataTrainingArguments, EvalArguments

writer = SummaryWriter("runs/exp1")

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 8
# EPOCHS = 1
LEARNING_RATE = 1e-05

## Loading Components
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, Seq2SeqTrainingArguments))
model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file="train.json")

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name,
    cache_dir=model_args.cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
    return_tensors='tf', padding=True,
    cache_dir=model_args.cache_dir,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name,
    config=config,
    cache_dir=model_args.cache_dir,
)
model.train()

## Reading data
data = pd.read_csv("../datasets/cards_waterloo.csv", low_memory=False)

train_dataset = ClaimsData(data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN)
valid_dataset = ClaimsData(data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN)
test_dataset = ClaimsData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN)


# Training
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
)

trainer.train()
trainer.save_model(Path(training_args.output_dir).joinpath("best-epoch")) #save best epoch