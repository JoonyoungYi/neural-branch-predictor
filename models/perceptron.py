from collections import deque


class _LocalPerceptron:
    weights = []
    N = 0
    bias = 0
    threshold = 0

    def __init__(self, N):
        self.N = N
        self.bias = 0
        self.threshold = 2 * N + 14  # optimal threshold depends on history length
        self.weights = [0] * N

    def predict(self, traces):
        running_sum = self.bias
        for i in range(
                0, self.N):  # dot product of trace history with the weights
            running_sum += traces[i] * self.weights[i]
        prediction = -1 if running_sum < 0 else 1
        return (prediction, running_sum)

    def train(self, prediction, actual, traces, running_sum):
        if (prediction != actual) or (abs(running_sum) < self.threshold):
            self.bias = self.bias + (1 * actual)
            for i in range(0, self.N):
                self.weights[i] = self.weights[i] + (actual * traces[i])

    def statistics(self):
        print("bias is: " + str(self.bias) + " weights are: " +
              str(self.weights))


class PerceptronPredictor:
    def __init__(self, N):
        traces = deque([])
        traces.extend([0] * N)

        self.traces = traces
        self.local_perceptron_dict = {}
        self.N = N

    def _get_local_perceptron(self, register):
        local_perceptron_dict = self.local_perceptron_dict
        return local_perceptron_dict.get(register, _LocalPerceptron(self.N))

    def _put_local_perceptron(self, register, local_perceptron):
        self.local_perceptron_dict[register] = local_perceptron

    def train(self, register, taken):
        traces = self.traces

        local_perceptron = self._get_local_perceptron(register)
        taken_pred, running_sum = local_perceptron.predict(traces)

        local_perceptron.train(taken_pred, 1
                               if taken else -1, traces, running_sum)

        traces.appendleft(1 if taken else -1)
        traces.pop()

        self._put_local_perceptron(register, local_perceptron)
        self.traces = traces

    def predict(self, register):
        traces = self.traces

        local_perceptron = self._get_local_perceptron(register)
        taken_pred, running_sum = local_perceptron.predict(traces)
        return max(taken_pred, 0)
