import pickle
from torch.nn.functional import one_hot
from sklearn.preprocessing import LabelEncoder
from utils import denoise_text

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

class ClaimsData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, device, eval=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.device = device
        self.eval = eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get the sample indexed from the dataset"""

        text = str(self.data.loc[index, "text"])
        
        ## Preprocessing
        text = denoise_text(text)

        ## Tokenizer
        tokenized_text = self.tokenizer( text, add_special_tokens = True, truncation=True,
            max_length = self.max_len, padding = "max_length", return_token_type_ids = True,
            return_tensors = "pt"
        ).to(self.device)  

        tokenized_text = {k:torch.squeeze(tokenized_text[k], 0) for k in tokenized_text}

        if not self.eval:
            label = int(self.data.loc[index, "labels"])
            tokenized_text["label"] = label

        return tokenized_text
    

class TaxonomyData(ClaimsData):
    def __init__(self, dataframe, tokenizer, max_len, num_classes, device, eval=False):
        ClaimsData.__init__(self, dataframe, tokenizer, max_len, device, eval)

        self.num_classes = num_classes
        with open('../cards/models/label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

    def __getitem__(self, index):
        """Get the sample indexed from the dataset"""

        tokenized_text = ClaimsData.__getitem__(self, index)

        if not self.eval:
            label = self.data.loc[index, "claim"]
            tokenized_text["label"] = one_hot(torch.tensor(self.le.transform([label])[0]-1), self.num_classes).float()

        return tokenized_text

class FileDataset(IterableDataset):
    def __init__(self, data_file, tokenizer, max_len, device, initial_pointer=0):
        self.data_file = data_file
        self.length = self.compute_length()
        self.initial_pointer = initial_pointer + 1
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        
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

        ## Tokenizer
        tokenized_text = self.tokenizer( text, add_special_tokens = True, truncation=True,
            max_length = self.max_len, padding = "max_length", return_token_type_ids = True,
            return_tensors = "pt"
        ).to(self.device)  

        tokenized_text = {k:torch.squeeze(tokenized_text[k], 0) for k in tokenized_text}
        tokenized_text["id"] = torch.tensor(int(id_))
        return tokenized_text

    def __iter__(self):
        file = open(self.data_file)
        i = 0
        while i < self.initial_pointer:
            file.readline()
            i += 1
        iter_map = map(self.preprocess_text, file)
        return iter_map
