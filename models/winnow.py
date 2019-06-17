import math
from collections import deque


class _LocalWinnow:
    weights = []
    N = 0
    bias = 0
    threshold = 0

    def __init__(self, N):
        self.base = math.e
        # self.base = 3
        # self.base = 2.7
        self.N = N
        self.bias = 1
        # self.threshold = 0
        # self.threshold = N  # optimal threshold depends on history length
        self.weights = [1] * N

    def predict(self, traces):
        running_sum = self.bias

        for i in range(0, self.N):
            # dot product of trace history with the weights
            trace = traces[i]
            running_sum += trace * self.weights[i]

        prediction = 0 if running_sum < (self.N / 2) else 1
        return (prediction, running_sum)

    def train(self, prediction, actual, traces, running_sum):
        if (prediction != actual) or (self.N / self.base < running_sum <
                                      self.base * self.N):
            if actual == 1:
                self.bias = self.bias * self.base
            else:
                self.bias = self.bias / self.base
            # self.bias = self.bias * math.pow(self.base, actual)

            for i in range(0, self.N):
                trace = traces[i]
                if trace != 1:
                    continue

                if actual == 1:
                    self.weights[i] = self.weights[i] * self.base
                else:
                    self.weights[i] = self.weights[i] / self.base

    def statistics(self):
        print("bias is: " + str(self.bias) + " weights are: " +
              str(self.weights))


class WinnowPredictor:
    def __init__(self, N):
        traces = deque([], maxlen=N)
        traces.extend([1] * N)

        self.traces = traces
        self.local_winnow_dict = {}
        self.N = N

    def _get_local_winnow(self, register):
        local_winnow_dict = self.local_winnow_dict
        return local_winnow_dict.get(register, _LocalWinnow(self.N))

    def _put_local_winnow(self, register, local_winnow):
        self.local_winnow_dict[register] = local_winnow

    def train(self, register, taken):
        traces = self.traces

        local_winnow = self._get_local_winnow(register)
        taken_pred, running_sum = local_winnow.predict(traces)

        local_winnow.train(taken_pred, 1 if taken else 0, traces, running_sum)

        traces.appendleft(1 if taken else 0)

        self._put_local_winnow(register, local_winnow)
        self.traces = traces

    def predict(self, register):
        traces = self.traces

        local_winnow = self._get_local_winnow(register)
        taken_pred, running_sum = local_winnow.predict(traces)
        return max(taken_pred, 0)
