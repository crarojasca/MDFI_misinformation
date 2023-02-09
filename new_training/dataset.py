from utils import denoise_text

import torch
from torch.utils.data import Dataset, DataLoader

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

        text = str(self.data.loc[index, "prompt"])
        
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