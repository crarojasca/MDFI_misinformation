import os
import math
import pickle
import argparse
import pandas as pd

from tqdm import tqdm
from scipy.special import softmax
from utils import preprocess_text

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel

from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

parser = argparse.ArgumentParser(description="Training SimpleTransformers Pipe")
parser.add_argument('--dataset', default='all', help='all - cards - waterloo')
parser.add_argument('--model_type', default='roberta', help='The type of model to use.')
parser.add_argument('--model_name', default='roberta_large', help='The exact architecture and trained weights to use.')
args = parser.parse_args()

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use GPU {}:'.format(
        torch.cuda.current_device()), torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


## Reading data
data = pd.read_csv("datasets/cards_waterloo.csv", low_memory=False)

## Selecting Dataset
if args.dataset == "all":
    pass # Use the full dataset
elif args.dataset == "cards":
    data = data[data["DATASET"] == "cards"]
elif args.dataset == "waterloo":
    data = data[data["DATASET"] == "waterloo"]
else:
    raise(f"The {args.dataset} dataset option is not supported.")


## Preprocess Texts
data["text"] = preprocess_text(data["prompt"])

## Encode labels
# Load label encoder
data = data.rename(columns={"completion" : "labels"})


## Partition Dataset
train = data[data.PARTITION=="TRAIN"]
valid = data[data.PARTITION=="VALID"]
test = data[data.PARTITION=="TEST"]


# Define additional model performance scores (F1)
def f1_multiclass_macro(labels, preds):
    return f1_score(labels, preds, average='macro')
def f1_multiclass_micro(labels, preds):
    return f1_score(labels, preds, average='micro')
def f1_multiclass_weighted(labels, preds):
    return f1_score(labels, preds, average='weighted')
def f1_class(labels, preds):
    return f1_score(labels, preds, average=None)
def precision(labels, preds):
    return precision_score(labels, preds, average='macro')
def recall(labels, preds):
    return recall_score(labels, preds, average='macro')

# Check the distribution of categories
print(round(train.labels.value_counts(normalize=True),2))
# Calculate weights
weights = compute_class_weight(
    class_weight='balanced', 
    classes=train.labels.unique(), 
    y=train.labels
)
weights = [*weights]
print(weights)


## TRAIN
# Create a ClassificationModel
model = ClassificationModel(args.model_type, args.model_name, 
                            num_labels = 2, weight = weights,
                            args={'reprocess_input_data': True, 
                                  'overwrite_output_dir': False,
                                  'output_dir': f'models/{args.dataset}-{args.model_type}/',
                                  'best_model_dir': f'models/{args.dataset}-{args.model_type}/best_model/',
                                  # Hyperparameters
                                  'train_batch_size': 8,
                                  'num_train_epochs': 4, 
                                  'learning_rate': 1e-5,
                                  # Text processing
                                  'max_seq_length': 256,
                                  'sliding_window': True,
                                  'stride': 0.6,
                                  'do_lower_case': False,
                                  # Evaluation
                                  'evaluate_during_training': True,
                                  'evaluate_during_training_verbose': True,
                                  'evaluate_during_training_steps': -1,
                                  # Saving
                                  'save_model_every_epoch': True,
                                  'save_eval_checkpoints': True,
                                  'weight_decay': 0
                                  })

# Train and evaluate the model
model.train_model(train, eval_df = valid,
                  f1_macro = f1_multiclass_macro, 
                  f1_micro = f1_multiclass_micro, 
                  f1_weighted = f1_multiclass_weighted, 
                  acc = accuracy_score, 
                  f1_class = f1_class)

## EVALUATE
## Valid
# Evaluate the classifier performance on the validation data
result, model_outputs, wrong_predictions = model.eval_model(valid, 
                                                            f1_macro = f1_multiclass_macro,
                                                            precision = precision, 
                                                            recall = recall,
                                                            acc = accuracy_score,
                                                            f1_micro = f1_multiclass_micro, 
                                                            f1_weighted = f1_multiclass_weighted, 
                                                            f1_class = f1_class)

print('\n\nThese are the results when testing the model on the validation data set:\n')
print(result)

# Evaluate the classifier performance on the testing data
result_test, model_outputs_test, wrong_predictions_test = model.eval_model(test, 
                                                                           f1_macro = f1_multiclass_macro,
                                                                           precision = precision, 
                                                                           recall = recall,
                                                                           acc = accuracy_score,
                                                                           f1_micro = f1_multiclass_micro, 
                                                                           f1_weighted = f1_multiclass_weighted,
                                                                           f1_class = f1_class)
print('\n\nThese are the results when testing the model on the testing data set:\n')
print(result_test)