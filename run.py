import pdb
import os
from collections import deque

from models.saturating_counter import SaturatingCounter
from dataset import Dataset
from configs import TEST_SPLIT


def __get_prediction_accuracy(model, traces):
    train_traces = traces[:int(len(traces) * (1 - TEST_SPLIT))]
    test_traces = traces[int(len(traces) * (1 - TEST_SPLIT)):]

    for register, taken in train_traces:
        model.train(register, taken)

    n_correct = 0
    for register, taken in test_traces:
        _taken = model.predict(register)
        if _taken == taken:
            n_correct += 1
        model.train(register, taken)
    return n_correct / len(test_traces)


def _simulate(args):
    traces = Dataset(args.dataset_idx).traces
    model = SaturatingCounter()
    accuracy = __get_prediction_accuracy(model, traces)
    print('%.5f' % accuracy)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='neural-branch-predictor')
    parser.add_argument(
        '--dataset_idx', type=int, default=1, help='dataset_idx')
    args = parser.parse_args()
    _simulate(args)


if __name__ == '__main__':
    main()
