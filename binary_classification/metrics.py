import torch
import evaluate
import numpy as np

metrics = ["accuracy", "roc_auc"]

def compute_metrics(eval_pred):    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    prediction_scores = np.max(torch.tensor(logits).softmax(dim=1).numpy(), axis=1)

    # return computed_metrics
    return evaluate.load("roc_auc").compute(prediction_scores=prediction_scores, references=labels)