from collections import deque

import torch
import torch.nn as nn
from torch.autograd import Variable


# Neural Network Model (1 hidden layer)
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MultiLayerPerceptronPredictor:
    def __init__(self, N):
        self.is_offline = True

        net = MultiLayerPerceptron(N, N, 2)
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        self.net = net
        self.criterion = criterion
        self.N = N
        self.optimizer = optimizer
        self.local_trace_dict = {}

        traces = self._make_initial_traces()
        self.traces = traces

    def _make_initial_traces(self):
        N = self.N

        traces = deque([])
        traces.extend([0] * N)
        return traces

    def _get_local_traces(self, register):
        return self.local_trace_dict.get(register, self._make_initial_traces())

    def _put_local_traces(self, register, traces):
        self.local_trace_dict[register] = traces

    def _get_global_traces(self):
        return self.traces

    def _put_global_traces(self, traces):
        self.traces = traces

    def train(self, register, taken):
        # traces = self._get_global_traces()
        traces = self._get_local_traces(register)

        optimizer = self.optimizer
        net = self.net
        criterion = self.criterion

        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(torch.FloatTensor([list(traces)]).cuda())
        loss = criterion(outputs,
                         torch.LongTensor([1] if taken else [0]).cuda())
        loss.backward()
        optimizer.step()

        traces.appendleft(1 if taken else -1)
        traces.pop()

        # self._put_global_traces(traces)
        self._put_local_traces(register, traces)

    def predict(self, register):
        traces = self._get_local_traces(register)
        net = self.net

        outputs = net(torch.FloatTensor([list(traces)]).cuda())
        o = outputs.tolist()[0]
        return 1 if o[0] > o[1] else 0

    def ready(self, register, taken):
        traces = self._get_local_traces(register)
        traces.appendleft(1 if taken else -1)
        traces.pop()
        self._put_local_traces(register, traces)
