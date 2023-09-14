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

from dataset import ClaimsData, TaxonomyData
from metrics import compute_metrics
from arguments import (
    ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments
)

writer = SummaryWriter("runs/exp1")

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
# EPOCHS = 1
LEARNING_RATE = 1e-05

device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_file = "train_hcards_5.3.json"
## Loading Components
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments))
model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file=config_file)

print( model_args.model_name)
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name,
    cache_dir = model_args.cache_dir,
    num_labels = model_args.num_labels,
    problem_type = model_args.problem_type
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
model.to(device)
model.train()

## Reading data
data = pd.read_csv(data_args.data_dir, low_memory=False)
if "hcards_5.3" in data_args.data_dir or "hcards_complete" in data_args.data_dir:
    data = data[(data.claim!="0_0")].copy(deep=True)

# train_dataset = ClaimsData(data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN, device)
# valid_dataset = ClaimsData(data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN, device)
# test_dataset = ClaimsData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, device)

train_dataset = TaxonomyData(data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
valid_dataset = TaxonomyData(data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
test_dataset = TaxonomyData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)


# Training
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(Path(training_args.output_dir).joinpath(model_args.save_name)) #save best epoch
