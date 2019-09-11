import torch
import random
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data_file, seq_len, is_train=True, train_size=0.7):
        with open(data_file, 'r') as f:
            self.data_x = f.read()
            self.data_x = [:int(train_size * len(self.data_x))] if is_train else [int(train_size * len(self.data_x)):]

        self.data_y = self.data_x[1:] + self.data_x[0]
        
        self.vocab = list(set(self.data_x))
        self.vocab_2_index = {k:v for v,k in enumerate(self.vocab)}
        self.index_2_vocab = {k:v for k,v in enumerate(self.vocab)}

        self.seq_len = seq_len
        self.indices = [i for i in range(len(self.data_x)/seq_len)]
        if is_train:
            random.shuffle(self.indices)

    def get_vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.indices)

    def char_to_onehot(self, character):
        embedding = torch.zeros(self.get_vocab_size())
        embedding[self.vocab_2_index[character]] = 1
        return embedding.vew(1, -1)

    def __getitem__(self, index):
        start = index * seq_len
        end = min((index + 1) * seq_len, len(self.data_file))
        
        text_x = self.data_x[start:end]

        text_x = [self.char_to_onehot(c) for c in text]
        text_x = torch.cat(text, 0)

        text_y = self.data_y[start:end]
        text_y = [self.vocab_2_index[c] for c in text_y]

        return text_x, torch.tensor(text_y)




    
