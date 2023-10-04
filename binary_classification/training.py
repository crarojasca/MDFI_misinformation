# Importing the libraries needed
import pickle
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

from dataset import BinaryDataset, TaxonomyData
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

MODEL_NAME = "waterloo_cards"


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


## Loading Components
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments))
model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file="train.json")

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
    # cache_dir=model_args.cache_dir,
)
# tokenizer.to(device)

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name,
    config=config,
    # cache_dir=model_args.cache_dir,
)
model.to(device)
model.train()

## Reading data
data = pd.read_csv(data_args.data_dir, low_memory=False)

print(data.groupby(["DATASET", "labels"])["text"].count())
print(data["PARTITION"].value_counts())

# data = data[data.DATASET=="cards"]
# data = data[(data.DATASET=="cards")&(data.claim!="0_0")].copy(deep=True)

# train_dataset = ClaimsData(data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN, device)
# valid_dataset = ClaimsData(data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN, device)
# test_dataset = ClaimsData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, device)

train_dataset = BinaryDataset(
    dataframe=data[data["PARTITION"] == "TRAIN"].reset_index(), 
    tokenizer=tokenizer, max_len=MAX_LEN, 
    # num_classes=model_args.num_labels, 
    device=device)
valid_dataset = BinaryDataset(
    data[data["PARTITION"] == "VALID"].reset_index(), 
    tokenizer=tokenizer, max_len=MAX_LEN, 
    # num_classes=model_args.num_labels, 
    device=device)
test_dataset = BinaryDataset(
    data[data["PARTITION"] == "TEST"].reset_index(), 
    tokenizer=tokenizer, max_len=MAX_LEN, 
    # num_classes=model_args.num_labels, 
    device=device)

print(train_dataset[0].keys())
print(valid_dataset[0].keys())

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


dataset = BinaryDataset(data, tokenizer, MAX_LEN, model_args.num_labels, device)
# with open('../cards/models/label_encoder.pkl', 'rb') as f:
#     le = pickle.load(f)

predictions = []
scores = []
for batch in tqdm(dataset):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    # predictions += le.inverse_transform(prediction.to('cpu') + 1).tolist()
    prediction += prediction.to('cpu').tolist()
    scores += score.tolist()

data[f"{MODEL_NAME}_pred"] = predictions
data[f"{MODEL_NAME}_proba"] = scores

# data.loc[data.DATASET=="GPT3-generated", f"{MODEL_NAME}_pred_parallel"] = predictions
# data.loc[data.DATASET=="GPT3-generated", f"{MODEL_NAME}_proba_parallel"] = [str(s) for s in scores]

data.to_csv(data_args.data_dir, index=False)