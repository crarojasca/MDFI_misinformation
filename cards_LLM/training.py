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

from sklearn.preprocessing import LabelEncoder


from dataset import TaxonomyData
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

device = "cuda" if torch.cuda.is_available() else "cpu"

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

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name,
    config=config,
    # cache_dir=model_args.cache_dir,
)
model.to("cuda")
model.train()

FILE = "cards_augmented_0_V1.csv"
## Reading data
<<<<<<< HEAD:cards_LLM/training.py
data = pd.read_csv(data_args.data_dir + FILE, low_memory=False)
data = data[data.claim!="0_0"]

with open('../cards/models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

le = LabelEncoder()
le.fit(data["claim"])
=======
data = pd.read_csv(data_args.data_dir, low_memory=False)
# data = data[data.DATASET=="cards"]
# data = data[(data.DATASET=="cards")&(data.claim!="0_0")].copy(deep=True)
>>>>>>> 95afb5f82e5d938dfb0e31d961fdc1040b402453:binary_classification/training.py

# train_dataset = ClaimsData(data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN, device)
# valid_dataset = ClaimsData(data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN, device)
# test_dataset = ClaimsData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, device)

<<<<<<< HEAD:cards_LLM/training.py
train_dataset = TaxonomyData(data[data["PARTITION"] == "TRAIN"].reset_index(), 
                             tokenizer, MAX_LEN, model_args.num_labels, le, device)
valid_dataset = TaxonomyData(data[data["PARTITION"] == "VALID"].reset_index(), 
                             tokenizer, MAX_LEN, model_args.num_labels, le, device)
# test_dataset = TaxonomyData(data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
=======
train_dataset = ClaimsData(
    data[data["PARTITION"] == "TRAIN"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
valid_dataset = ClaimsData(
    data[data["PARTITION"] == "VALID"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
test_dataset = ClaimsData(
    data[data["PARTITION"] == "TEST"].reset_index(), tokenizer, MAX_LEN, model_args.num_labels, device)
>>>>>>> 95afb5f82e5d938dfb0e31d961fdc1040b402453:binary_classification/training.py


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
<<<<<<< HEAD:cards_LLM/training.py
trainer.save_model(Path(training_args.output_dir).joinpath(model_args.save_name)) #save best epoch'


dataset = TaxonomyData(data, tokenizer, MAX_LEN, model_args.num_labels, le, device, eval=True)
dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


MODEL = "cards_aug"
predictions = []
scores = []
for batch in tqdm(dataloader):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    predictions += le.inverse_transform(prediction.to('cpu')).tolist()
    scores += [str(score.tolist())]

data[f"{MODEL}_pred"] = predictions
data[f"{MODEL}_proba"] = scores
data.to_csv("../datasets/cards_second_hierarchy.csv", index=False)
=======
trainer.save_model(Path(training_args.output_dir).joinpath(model_args.save_name)) #save best epoch

MODEL_NAME = "binary_roberta"
dataset = ClaimsData(data, tokenizer, MAX_LEN, model_args.num_labels, device)
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
>>>>>>> 95afb5f82e5d938dfb0e31d961fdc1040b402453:binary_classification/training.py
