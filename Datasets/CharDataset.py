import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CustomCollate(object):
    def __init__(self):
        super(CustomCollate, self).__init__()

    def custom_collate(self, batch):
        batch_X = [torch.tensor(elem[0]) for elem in batch]
        batch_Y = [torch.tensor(elem[1]) for elem in batch]
        batch_lengths = torch.tensor([elem[2] for elem in batch])

        sorted_indices = torch.argsort(batch_lengths)
        sorted_X = [batch_X[i.item()] for i in sorted_indices]
        sorted_Y = [batch_Y[i.item()] for i in sorted_indices]
        sorted_lengths = batch_lengths[sorted_indices]

        return (pad_sequence(sorted_X), pad_sequence(sorted_Y), sorted_lengths)

    def __call__(self, batch):
        self.custom_collate(self, batch)


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
