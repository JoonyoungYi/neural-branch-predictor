import pdb
import os
from collections import deque
from time import time
import numpy as np

def sim_new(pred, file='mcf_result', **kwargs):
    trace = {}
    branches = []
    i = 0
    with open(file, 'r') as file_in:
        for line in file_in:
            if len(line) != 11:
                continue
            register = line[0:8]
            result = int(line[9])
            trace.setdefault(register, []).append(result)
            branches.append([register, result])
            i += 1
            if i >= int(1e+7):
                break

    start_time = time()
    num_correct = pred(branches, l=kwargs['l'])
    end_time = time()
    time_len = end_time - start_time
    total = sum(len(r) for r in trace.values())
    return (num_correct * 1.0/total), time_len


class Counter:
    state = 2   # 1 and 2 predict do not take, 3 and 4 predict take
    def predict(self):
        if(self.state < 3):
            return -1
        if(self.state > 2):
            return 1

    def update(self, actual):
        if(actual == 1):
            self.state = self.state + 1
            if(self.state > 4):
                self.state = 4
        if(actual == -1):
            self.state = self.state - 1
            if(self.state < 1):
                self.state = 1
        return


def saturating_counter(trace, l=1):

    c_list = {}
    num_correct = 0

    for br in trace:            # iterating through each branch
        if br[0] not in c_list:     # if no previous branch from this memory location
            c_list[br[0]] = Counter()
        pr = c_list[br[0]].predict()
        actual_value = 1 if br[1] else -1
        c_list[br[0]].update(actual_value)
        if pr == actual_value:
            num_correct += 1
    return num_correct

class Perceptron:

    def __init__(self, N):
        self.N = N
        self.bias = 0
        self.threshold = 2 * N + 14                 # optimal threshold depends on history length
        self.weights = [0] * N

    def predict(self, global_branch_history):
        running_sum = self.bias
        for i in range(0, self.N):                  # dot product of branch history with the weights
            running_sum += global_branch_history[i] * self.weights[i]
        prediction = -1 if running_sum < 0 else 1
        return (prediction, running_sum)

    def update(self, prediction, actual, global_branch_history, running_sum):
        if (prediction != actual) or (abs(running_sum) < self.threshold):
            self.bias = self.bias + (1 * actual)
            for i in range(0, self.N):
                self.weights[i] = self.weights[i] + (actual * global_branch_history[i])

    def statistics(self):
        print "bias is: " + str(self.bias) + " weights are: " + str(self.weights)

def perceptron_pred(trace, l=1):

    global_branch_history = deque([])
    global_branch_history.extend([0]*l)

    p_list = {}
    num_correct = 0

    for br in trace:            # iterating through each branch
        if br[0] not in p_list:     # if no previous branch from this memory location
            p_list[br[0]] = Perceptron(l)
        results = p_list[br[0]].predict(global_branch_history)
        pr = results[0]
        running_sum = results [1]
        actual_value = 1 if br[1] else -1
        p_list[br[0]].update(pr, actual_value, global_branch_history, running_sum)
        global_branch_history.appendleft(actual_value)
        global_branch_history.pop()
        if pr == actual_value:
            num_correct += 1

    return num_correct

class Winnow:

    def __init__(self, N):
        self.base = 2
        self.N = N
        self.weights = [1] * N
        self.bias = 1
        self.threshold = N

    def predict(self, global_branch_history):
        running_sum = self.bias
        for i in range(0, self.N):                  # dot product of branch history with the weights
            running_sum += global_branch_history[i] * self.weights[i]
        prediction = -1 if (running_sum < self.N) else 1
        return (prediction, running_sum)

    def update(self, prediction, actual, global_branch_history, running_sum):
        if (prediction != actual):
            self.bias = self.bias * (self.base**actual)
            for i in range(0, self.N):
                self.weights[i] = self.weights[i] * (self.base**(actual * global_branch_history[i]))

    def statistics(self):
        print "bias is: " + str(self.bias) + " weights are: " + str(self.weights)

def winnow_pred(trace, l=1):

    global_branch_history = deque([])
    global_branch_history.extend([0]*l)

    p_list = {}
    num_correct = 0

    for br in trace:            # iterating through each branch
        if br[0] not in p_list:     # if no previous branch from this memory location
            p_list[br[0]] = Winnow(l)
        results = p_list[br[0]].predict(global_branch_history)
        pr = results[0]
        running_sum = results [1]
        actual_value = 1 if br[1] else -1
        p_list[br[0]].update(pr, actual_value, global_branch_history, running_sum)
        global_branch_history.appendleft(actual_value)
        global_branch_history.pop()
        if pr == actual_value:
            num_correct += 1

    return num_correct



def main():

    method_list = [winnow_pred]
    len_list = [8, 16, 32, 64]
    data_list = ['bzip2_result.out', 'gcc_result.out', 'gzip_result.out', 'mcf_result.out', 'parser_result.out', 'twolf_result.out', 'vortex_result.out', 'vpr_result.out']

    for data in data_list:
        for method in method_list:
            if method == saturating_counter:
                result, time_len = sim_new(method, file=data, l=8)
                print "%s   %s  %d  %.5f    %.5f" % (data.split('_')[0] ,method.__name__, 8, result, time_len)
            else:
                for length in len_list:
                    result, time_len = sim_new(method, file=data, l=length)
                    print "%s   %s  %d  %.5f    %.5f" % (data.split('_')[0] ,method.__name__, length, result, time_len)

if __name__ == '__main__':
    main()
