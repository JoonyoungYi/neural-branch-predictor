class Perceptron:
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

    def update(self, prediction, actual, global_trace_history, running_sum):
        if (prediction != actual) or (abs(running_sum) < self.threshold):
            self.bias = self.bias + (1 * actual)
            for i in range(0, self.N):
                self.weights[
                    i] = self.weights[i] + (actual * global_trace_history[i])

    def statistics(self):
        print("bias is: " + str(self.bias) + " weights are: " +
              str(self.weights))


# perceptron_pred
#   @params
#       trace: a list that contains pairs of tracee memory locations and whether the trace was taken or not
#           ex: ['47aa6d', 1]
#       l: length of the trace history to use in making decisions
#   @returns
#       num_correct: the number of traces correctly predicted
#


def perceptron_pred(trace, l=1):

    global_trace_history = deque([])
    global_trace_history.extend([0] * l)

    p_list = {}
    num_correct = 0

    for br in trace:  # iterating through each trace
        if br[0] not in p_list:  # if no previous trace from this memory location
            p_list[br[0]] = Perceptron(l)
        results = p_list[br[0]].predict(global_trace_history)
        pr = results[0]
        running_sum = results[1]
        actual_value = 1 if br[1] else -1
        p_list[br[0]].update(pr, actual_value, global_trace_history,
                             running_sum)
        global_trace_history.appendleft(actual_value)
        global_trace_history.pop()
        if pr == actual_value:
            num_correct += 1

    return num_correct
