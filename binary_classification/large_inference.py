import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from accelerate import Accelerator

from dataset import FileDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FILES
## Input File
DATA_FILE = "../datasets/omm_export_tweets_01-06-2022.csv" ## Hamburg

## Output File
PREDICTIONS_FILE = "../datasets/hamburg_aug_xlnet_predictions.txt"

MAX_LEN = 256
BATCH_SIZE = 8
MODEL_NAME = "aug_xlnet"
MODEL_DIR = "experiments/results/aug_xlnet/best-epoch"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model
print("Loading model: {}".format(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)

# Load Number lines Predictions
with open(PREDICTIONS_FILE, "rb") as f:
    last_line = sum(1 for line in f if line.rstrip())

# Load Dataset
dataset = FileDataset(DATA_FILE, tokenizer=tokenizer, max_len=MAX_LEN, device=device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize accelerate
accelerator = Accelerator()
model, dataloader = accelerator.prepare(
    model, dataloader
)


with open(PREDICTIONS_FILE, "a") as f:

    for batch in tqdm(dataloader, initial=last_line//2000):
        outputs = model(**batch)
        prediction = torch.argmax(outputs.logits, axis=1).tolist()
        scores = outputs.logits.softmax(dim = 1).tolist()
        ids = batch["id"].tolist()
        
        preds_batch = ["|".join(map(str, pred)) for pred in zip(ids, prediction, scores)]
        preds_batch = "\n".join(preds_batch) + "\n"
        f.write(preds_batch)