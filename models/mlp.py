from collections import deque


class MultiLayerPerceptronPredictor:
    def __init__(self, N):
        traces = deque([])
        traces.extend([0] * N)

        self.traces = traces
        self.N = N

    def train(self, register, taken):
        traces = self.traces

        traces.appendleft(1 if taken else -1)
        traces.pop()
        self.traces = traces

    def predict(self, register):
        # traces = self.traces
        return 1
