import re
import torch
import unicodedata
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

# Define the model 
architecture = 'roberta'
# model_name = 'CARDS_RoBERTa_Classifier'
model_name = "cards/models/CARDS_RoBERTa_Classifier"

# Load the classifier
roberta_model = ClassificationModel(architecture, model_name)

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

def preprocess_text(column):
    return column.astype(str).apply(denoise_text)

