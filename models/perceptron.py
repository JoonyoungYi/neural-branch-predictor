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

    def predict(self, global_trace_history):
        running_sum = self.bias
        for i in range(
                0, self.N):  # dot product of trace history with the weights
            running_sum += global_trace_history[i] * self.weights[i]
        prediction = -1 if running_sum < 0 else 1
        return (prediction, running_sum)

    def train(self, prediction, actual, global_trace_history, running_sum):
        if (prediction != actual) or (abs(running_sum) < self.threshold):
            self.bias = self.bias + (1 * actual)
            for i in range(0, self.N):
                self.weights[
                    i] = self.weights[i] + (actual * global_trace_history[i])

    def statistics(self):
        print("bias is: " + str(self.bias) + " weights are: " +
              str(self.weights))


class Perceptron:
    def __init__(self, N):
        global_trace_history = deque([])
        global_trace_history.extend([0] * N)

        self.global_trace_history = global_trace_history
        self.local_perceptron_dict = {}
        self.N = N

    def _get_local_perceptron(self, register):
        local_perceptron_dict = self.local_perceptron_dict
        return local_perceptron_dict.get(register, _LocalPerceptron(self.N))

    def _put_local_perceptron(self, register, local_perceptron):
        self.local_perceptron_dict[register] = local_perceptron

    def train(self, register, taken):
        global_trace_history = self.global_trace_history

        local_perceptron = self._get_local_perceptron(register)
        taken_pred, running_sum = local_perceptron.predict(
            global_trace_history)

        local_perceptron.train(taken_pred, 1 if taken else -1, global_trace_history, running_sum)

        global_trace_history.appendleft(1 if taken else -1)
        global_trace_history.pop()

        self._put_local_perceptron(register, local_perceptron)
        self.global_trace_history = global_trace_history

    def predict(self, register):
        global_trace_history = self.global_trace_history

        local_perceptron = self._get_local_perceptron(register)
        taken_pred, running_sum = local_perceptron.predict(
            global_trace_history)
        return taken_pred
