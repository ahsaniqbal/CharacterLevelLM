import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CustomCollate(object):
    def __init__(self):
        super(CustomCollate, self).__init__()

    def get_mask(self, max_length, length):
        mask = np.zeros(max_length, dtype=np.float32)
        mask[:length] = 1
        return mask

    def custom_collate(self, batch):
        batch_X = [torch.tensor(elem[0]) for elem in batch]
        batch_Y = [torch.tensor(elem[1]) for elem in batch]
        batch_lengths = torch.tensor([elem[2] for elem in batch])
        sorted_indices = torch.argsort(batch_lengths, descending=True)
        sorted_X = [batch_X[i.item()] for i in sorted_indices]
        sorted_Y = [batch_Y[i.item()] for i in sorted_indices]
        sorted_lengths = batch_lengths[sorted_indices]
        mask = list(map(lambda x: torch.ones(x), sorted_lengths))
        return (pad_sequence(sorted_X), pad_sequence(sorted_Y), pad_sequence(mask))

    def __call__(self, batch):
        return self.custom_collate(batch)


class CharDataset(Dataset):
    def __init__(self, data_file, seq_len,
                 is_train=True, train_size=0.7):
        with open(data_file, 'r') as f:
            self.data_x = f.read()
            if is_train:
                self.data_x = self.data_x[:int(train_size * len(self.data_x))]
            else:
                self.data_x = self.data_x[int(train_size * len(self.data_x)):]

        self.data_y = self.data_x[1:] + self.data_x[0]
        self.vocab = list(set(self.data_x))
        self.vocab_2_index = {k: v for v, k in enumerate(self.vocab)}
        self.index_2_vocab = {k: v for k, v in enumerate(self.vocab)}

        self.seq_len = seq_len
        self.indices = [i for i in range(int(len(self.data_x) // seq_len)) ]
        if is_train:
            random.shuffle(self.indices)

    def get_vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.indices)

    def char_to_onehot(self, character):
        embedding = np.zeros(self.get_vocab_size(), dtype=np.float32)
        embedding[self.vocab_2_index[character]] = 1
        return embedding

    def __getitem__(self, index):
        start = index * self.seq_len
        end = min((index + 1) * self.seq_len, len(self.data_x))
        text_x = self.data_x[start:end]

        text_x = np.array([self.char_to_onehot(c) for c in text_x])

        text_y = self.data_y[start:end]
        text_y = np.array([self.vocab_2_index[c] for c in text_y])
        
        return (text_x, text_y, len(text_x))
