import re
import os
import time
import pickle
import unicodedata
import pandas as pd
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch


if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use GPU {}:'.format(
        torch.cuda.current_device()), 
        torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Define required functions

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

# Define text pre-processing functions
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
def strip_underscores(text):
    return re.sub(r'_+', ' ', text)
def remove_multiple_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)

# Merge text pre-processing functions
def denoise_text(text):
    text = remove_between_square_brackets(text)
    text = remove_non_ascii(text)
    text = strip_underscores(text)
    text = remove_multiple_spaces(text)
    return text.strip()

# Load label encoder
with open('../cards/models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

samples = 50
VERSION = "V4"
augment = True
V2 = False

# Load and pre-process the text data
# Load the data
train = pd.read_csv('../cards/data/training/training.csv')
train["PARTITION"] = "TRAIN"

# train_noclaim = train[train.claim=="0_0"][:1000]
# train = train[train.claim!="0_0"]
# train = pd.concat([train, train_noclaim], ignore_index=True)

valid = pd.read_csv('../cards/data/training/validation.csv')
valid["PARTITION"] = "VALID"
test = pd.read_csv('../cards/data/training/test.csv')
test["PARTITION"] = "TEST"
data = pd.concat([train, valid, test], ignore_index=True)
data["DATASET"] = "cards"



# Load augmented Data and format it
augmented = pd.read_csv('../datasets/generated_disinformation_taxonomy_CARDS_GPT-4_specific_samples_V1.csv')
augmented = augmented.rename(columns={"generated_label": "claim"})

augmentedv2 = pd.read_csv('../datasets/generated_disinformation_taxonomy_CARDS_CHATGPT_specific_samples_V2.csv')
augmentedv2 = augmentedv2.rename(columns={"generated_label": "claim"})
v2claims = augmentedv2.claim.unique()
print(v2claims)

sampled_augmented = pd.DataFrame()
if augment:
    for claim in augmented.claim.unique():

    # for claim in ['1_1', '1_2', '2_1', '4_4', '5_2']: 
        if claim in v2claims and V2:
            tmp = augmentedv2[(augmentedv2.claim==claim)].iloc[:samples, :]
            tmp["DATASET"] = "generated-chatgptV2"
        else:
            tmp = augmented[(augmented.claim==claim)].iloc[:samples, :]
            tmp["DATASET"] = "generated-chatgpt"

        sampled_augmented = pd.concat([sampled_augmented, tmp], ignore_index=True)


sampled_augmented["PARTITION"] = "TRAIN"

data = pd.concat([data, sampled_augmented], ignore_index=True)

# Pre-process the text
data['text'] = data['text'].astype(str).apply(denoise_text)

# Encode the labels
data['labels'] = le.fit_transform(data.claim)

print(data.groupby(["claim", "DATASET"]).text.count())
print(data.groupby(["PARTITION", "DATASET"]).text.count())

train = data[data.PARTITION=="TRAIN"].copy(deep=True)
valid = data[data.PARTITION=="VALID"].copy(deep=True)

# Check the distribution of categories
print(round(train.labels.value_counts(normalize=True), 2))
# Calculate weights
weights = compute_class_weight(
    class_weight='balanced', 
    classes=train.labels.unique(), 
    y=train.labels
)
weights = [*weights]
print(weights)
# print(torch.initial_seed())
# seed 0 400
#                       V1_0  V1_50 V2_50 V2_400
# 2891285791441657046   76.04       77.14 75.39
# 9834838408490912248   77.68 78.05 78.19 76.87
# 5941432209682881199   78.47       77.63 76.86
# 10415750228357744022  79.25 ----- ----- 77.98
# Create a ClassificationModel

for seed in [9834838408490912248]:
    PATH = f"../datasets/augmented/{seed}/"
    os.system(f"mkdir {PATH}")
    FILE = PATH + f"cards_augmented_{samples}_{VERSION}_SimCSE_base.csv"
    MODEL = "cards_aug"
    model = ClassificationModel('roberta', '../SimCSE/models/', 
                                num_labels = 18, weight = weights,
                                args={"manual_seed":seed % 2**32,
                                    'reprocess_input_data': True, 
                                    'overwrite_output_dir': False,
                                    'output_dir': f'models/{seed}/{MODEL}_{samples}_{VERSION}_SimCSE_base',
                                    'best_model_dir': f'models/{seed}/{MODEL}_{samples}_{VERSION}_SimCSE_base/best_model/',
                                    # Hyperparameters
                                    'train_batch_size': 6,
                                    'num_train_epochs': 3, 
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

    # model = ClassificationModel("roberta", f'models/{MODEL}{VERSION}/best_model/')
    # Predict the labels
    predictions, raw_outputs = model.predict(list(data.text.astype(str)))

    data[f'{MODEL}_pred'] = le.inverse_transform(predictions)
    data[f'{MODEL}_proba'] = [softmax(element[0]) for element in raw_outputs]
    data.to_csv(FILE, index=False)