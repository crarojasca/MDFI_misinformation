import os
import math
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.special import softmax
from utils import denoise_text, preprocess_text

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel

from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay)

from IPython.display import display, Markdown, Latex


if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use GPU {}:'.format(
        torch.cuda.current_device()), torch.cuda.get_device_name(torch.cuda.current_device()))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

## Load Data
data = pd.read_csv("datasets/hamburg_1000_sample.csv", low_memory=False)

## Preprocess Texts
data["text"] = preprocess_text(data["fulltext"])

## Encode labels
# Load label encoder
data = data.rename(columns={"completion" : "labels"})

## Original Model
# Load label encoder
with open('cards/models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
    
# Load model
model_location = "cards/models/CARDS_RoBERTa_Classifier"
model = "roberta"
roberta_model = ClassificationModel(model, model_location)

# Predict the labels
predictions, raw_outputs = roberta_model.predict(list(data.text.astype(str)))

data['cards_original_pred'] = le.inverse_transform(predictions)
data['cards_pred'] = data["cards_original_pred"].apply(lambda pred: 0 if pred=="0_0" else 1)
data['cards_proba'] = [softmax(element[0]) for element in raw_outputs]

## Waterloo Model
# Load Model
model_location = "models/waterloo"
model = "roberta"
roberta_model = ClassificationModel(model, model_location)

# Predict the labels
predictions, raw_outputs = roberta_model.predict(list(data.text.astype(str)))

data['waterloo_pred'] = predictions
data['waterloo_proba'] = [softmax(element[0]) for element in raw_outputs]


## Waterloo-CARDS Model
# Load Model
model_location = "models/waterloo-cards"
model = "roberta"
roberta_model = ClassificationModel(model, model_location)

# Predict the labels
predictions, raw_outputs = roberta_model.predict(list(data.text.astype(str)))

data['waterloo-cards_pred'] = predictions
data['waterloo-cards_proba'] = [softmax(element[0]) for element in raw_outputs]

data.to_csv("datasets/hamburg_1000_sample.csv", index=False)

