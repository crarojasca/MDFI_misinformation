from utils import denoise_text

from torch.utils.data import Dataset, DataLoader

class ClaimsData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, "prompt"])
        label = int(self.data.loc[index, "labels"])

        ## Preprocessing
        text = denoise_text(text)

        ## Tokenizer
        tokenized_text = self.tokenizer( text, add_special_tokens = True, truncation=True,
            max_length = self.max_len, padding = "max_length", return_token_type_ids = True
        )     
        tokenized_text["label"] = label

        return tokenized_text