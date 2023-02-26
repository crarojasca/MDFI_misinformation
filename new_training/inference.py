import os
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

# parser = argparse.ArgumentParser(description="Training SimpleTransformers Pipe")
# parser.add_argument('--model_name', default='roberta', help='The exact architecture and trained weights to use.')
# args = parser.parse_args()

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, TrainingArguments))
model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file="train.json")
model_dir = Path(training_args.output_dir).joinpath(model_args.save_name)


MAX_LEN = 256
BATCH_SIZE = 8
MODEL_NAME = os.path.basename(training_args.output_dir)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


print("Loading model: {}".format(model_dir))
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)


data = pd.read_csv(data_args.data_dir, low_memory=False)
dataset = ClaimsData(data, tokenizer=tokenizer, max_len=MAX_LEN, device=device, eval=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# # accelerator = Accelerator()
# # model, dataloader = accelerator.prepare(
# #     model, dataloader
# # )

predictions = []
scores = []
for batch in tqdm(dataloader):
    outputs = model(**batch)
    score = outputs.logits.softmax(dim = 1)
    prediction = torch.argmax(outputs.logits, axis=1)
    predictions += prediction.tolist()
    scores += score.tolist()

data[f"{MODEL_NAME}_pred"] = predictions
data[f"{MODEL_NAME}_proba"] = scores

data.to_csv(data_args.data_dir, index=False)