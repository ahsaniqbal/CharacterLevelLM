import torch
import torch.nn as nn
from torch.distributions import Categorical


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = self.fc(outputs.view(-1, outputs.shape[-1]))
        return outputs

    def sample(self, X):
        outputs, _ = self.lstm(X)
        outputs = self.fc(outputs.view(-1, outputs.shape[-1]))
        #m = Categorical(probs=self.softmax(outputs[-1].view(1, -1)))
        #return m.sample()
        return torch.argmax(outputs)[-1]

