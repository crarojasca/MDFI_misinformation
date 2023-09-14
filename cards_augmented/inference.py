import os
import pickle
import pandas as pd


from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from arguments import (
    ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments
)

from dataset import ClaimsData

import torch
from torch.utils.data import DataLoader

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments))
model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file="train.json")
model_dir = Path(training_args.output_dir).joinpath(model_args.save_name)

MAX_LEN = 256
BATCH_SIZE = 4
MODEL_NAME = os.path.basename(training_args.output_dir)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading model: {}".format(model_dir))
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)

print("Loading data: {}".format(data_args.data_dir))
data = pd.read_csv(data_args.data_dir, low_memory=False)
# selected_data = data[data.DATASET=="GPT3-generated"].copy(deep=True).reset_index()

dataset = ClaimsData(data, tokenizer=tokenizer, max_len=MAX_LEN, device=device, eval=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# # Initialize accelerate
# accelerator = Accelerator()
# model, dataloader = accelerator.prepare(
#     model, dataloader
# )

with open('../cards/models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

predictions = []
scores = []
for batch in tqdm(dataloader):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    predictions += le.inverse_transform(prediction.to('cpu') + 1).tolist()
    scores += score.tolist()

data[f"{MODEL_NAME}_pred"] = predictions
data[f"{MODEL_NAME}_proba"] = scores

# data.loc[data.DATASET=="GPT3-generated", f"{MODEL_NAME}_pred_parallel"] = predictions
# data.loc[data.DATASET=="GPT3-generated", f"{MODEL_NAME}_proba_parallel"] = [str(s) for s in scores]

data.to_csv(data_args.data_dir, index=False)