import os
import math
import pickle
import pandas as pd

from tqdm import tqdm
from scipy.special import softmax
from utils import denoise_text

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use GPU {}:'.format(
        torch.cuda.current_device()), torch.cuda.get_device_name(torch.cuda.current_device()))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# FILES
## Input File
data_file = "datasets/omm_export_tweets_01-06-2022.csv"
## Output File
predictions_file = "datasets/predictions_waterloo_cards"

model_location = "models/waterloo-cards"
model = "roberta"

roberta_model = ClassificationModel(model, model_location)
roberta_model.args.silent = True

class Dataset(IterableDataset):
    def __init__(self, data_file, initial_pointer=0):
        self.data_file = data_file
        self.length = self.compute_length()
        self.initial_pointer = initial_pointer + 1
        
    def compute_length(self):
        with open(self.data_file) as f:
            num_lines = sum(1 for line in f if line.rstrip())
        return num_lines
        
    def __len__(self):
        return self.length
    
    def preprocess_text(self, line):
        row = line.split("\t")
        id_ = row[0]
        text = denoise_text(row[2])
        return id_, text

    def __iter__(self):
        file = open(self.data_file)
        i = 0
        while i < self.initial_pointer:
            file.readline()
            i += 1
        iter_map = map(self.preprocess_text, file)
        return iter_map

# Load Number lines Predictions
with open(predictions_file, "rb") as f:
    last_line = sum(1 for line in f if line.rstrip())
    
dataset = Dataset(data_file, last_line)
dataloader = DataLoader(dataset, batch_size = 2000)

with open(predictions_file, "a") as f:

    for ids, texts in tqdm(dataloader, initial=last_line//2000):
        predictions_batch, raw_outputs_batch = roberta_model.predict(texts)
        scores_batch = [max(softmax(element[0])) for element in raw_outputs_batch]

        preds_batch = ["|".join(map(str, pred)) for pred in zip(ids, predictions_batch, scores_batch)]
        preds_batch = "\n".join(preds_batch) + "\n"
        f.write(preds_batch)