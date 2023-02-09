import argparse
import pandas as pd

from tqdm import tqdm
from dataset import ClaimsData
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Training SimpleTransformers Pipe")
parser.add_argument('--model_name', default='roberta', help='The exact architecture and trained weights to use.')
args = parser.parse_args()


MAX_LEN = 256
BATCH_SIZE = 4

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_dir = f"./experiments/results/{args.model_name}/best-epoch/"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)

print(f"Loaded model: {args.model_name}, from directory: {model_dir}")

data = pd.read_csv("../datasets/cards_waterloo.csv", low_memory=False)
dataset = ClaimsData(data, tokenizer=tokenizer, max_len=MAX_LEN, device=device, eval=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []
scores = []
for batch in tqdm(dataloader):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    predictions += prediction.tolist()
    scores += score.tolist()

data[f"{args.model_name}_pred"] = predictions
data[f"{args.model_name}_proba"] = scores

data.to_csv("../datasets/cards_waterloo.csv", index=False)