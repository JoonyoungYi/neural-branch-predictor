from collections import deque

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=1024,
            num_layers=3,
            batch_first=True,
            bias=True)
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        # print(x)
        h, c = self.lstm(x)
        h = h[:, 0, :]
        out = self.fc(h)
        return out
