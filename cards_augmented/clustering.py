import re
import torch
import pickle
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

from scipy.special import softmax
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel

import torch
from torch.utils.data import DataLoader


data = pd.read_csv("../datasets/augmented/9834838408490912248/cards_augmented_0_V1.csv")

with open('../cards/models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load and pre-process the text data
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

data["p_text"] = data.text.astype(str).apply(denoise_text)
data = data[data.claim!="0_0"].copy(deep=True)

# Define the model 
architecture = 'roberta'
# model_name = 'CARDS_RoBERTa_Classifier'
model_name = "../cards/models/CARDS_RoBERTa_Classifier"

# Load the classifier
roberta_model = ClassificationModel(architecture, model_name)
roberta_model.config.output_hidden_states = True

intervale = 500
n = len(data.p_text.tolist())
texts = data.p_text.tolist()
n_features = 262144
embeddings = np.empty([n, n_features])
for i in tqdm(range(0, n, intervale)):
    j = i+intervale if i+intervale<n else n
    print(i, j)
    _, _, all_embedding_outputs, _ = roberta_model.predict(texts[i : j])
    # embeddings[i:j, :] = np.zeros([j-i, n_features])
    embeddings[i:j, :] = all_embedding_outputs.reshape(all_embedding_outputs.shape[0], -1)
    # embeddings = np.concatenate((embeddings, emb_outputs), axis=0)
    # print(emb_outputs.shape)

pickle.dump(embeddings, open("embeddings.sav", 'wb'))

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

data[["x", "y"]] = tsne.fit_transform(embeddings)

fig = px.scatter(data, x="x", y="y", color="claim", facet_col="PARTITION")
fig.write_html("images/clustering.html")
data.to_csv("reduced_data.csv", index=True)