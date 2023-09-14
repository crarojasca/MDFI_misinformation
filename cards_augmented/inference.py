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
# MODEL_NAME = os.path.basename(training_args.output_dir)
MODEL_NAME = "cards_second_level_0_"
MODEL_DIR = "models/9834838408490912248/cards_second_level_0_"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading model: {}".format(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)


DIR = "../datasets/hamburg/"
FILE = "hamburg_joinsample.csv"
print("Loading data: {}".format(DIR))
data = pd.read_csv(DIR + FILE, low_memory=False)
# selected_data = data[data.DATASET=="GPT3-generated"].copy(deep=True).reset_index()

dataset = ClaimsData(data, tokenizer=tokenizer, max_len=MAX_LEN, device=device, eval=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

with open('label_encoder_second_level.pkl', 'rb') as f:
    le = pickle.load(f)

predictions = []
scores = []
for batch in tqdm(dataloader):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    predictions += le.inverse_transform(prediction.to('cpu')).tolist()
    scores += score.tolist()

data[f"cards_secondlvl_pred"] = predictions
data[f"cards_secondlvl_proba"] = scores

data.to_csv(DIR + FILE, index=False)