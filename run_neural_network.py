import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from tqdm import tqdm
except:
    print('There is no tqdm...')
    tqdm = lambda x: x

from dataset import Dataset
from configs import TEST_SPLIT


def ___init_q(n_history):
    q = deque(maxlen=n_history)
    for _ in range(n_history):
        q.append(0)
    return q


def ___get_histories(traces, n_history):

    # history_kind = 'global'
    history_kind = 'local'
    # history_kind = 'hybrid'

    if history_kind == 'global':
        q = ___init_q(n_history)
        histories = []
        for register, taken in traces:
            histories.append(list(q))
            q.append(taken)
        return histories
    elif history_kind == 'local':
        q_dict = {}
        histories = []
        for register, taken in traces:
            q = q_dict.get(register, ___init_q(n_history))
            histories.append(list(q))
            q.append(taken)
            q_dict[register] = q
        return histories
    elif history_kind == 'hybrid':
        q_global = ___init_q(n_history)
        q_dict = {}
        histories = []
        for register, taken in traces:
            q = q_dict.get(register, ___init_q(n_history))

            histories.append(list(q) + list(q_global))

            q_global.append(taken)
            q.append(taken)
            q_dict[register] = q
        return histories
    else:
        NotImplementedError()


def __get_prediction_accuracy(model, traces, args):
    print('>> start init histories...')
    x = ___get_histories(traces, args.n_history)
    print('>> end init histories.')
    y = [taken for _, taken in traces]

    offset = int(len(traces) * (1 - TEST_SPLIT))

    train_x = np.array(x[:offset])
    if args.model == 'lstm':
        train_x = np.expand_dims(train_x), axis=2)
    train_y = np.array(y[:offset])
    test_x = np.array(x[offset:])
    if args.model == 'lstm':
        test_x = np.expand_dims(test_x), axis=2)
    test_y = np.array(y[offset:])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    max_test_acc = 0
    max_epoch_idx = 0
    n_total = test_x.shape[0]

    n_batch = 1000

    for epoch_idx in range(10000):
        for start_idx in tqdm(range(0, train_x.shape[0], n_batch)):
            batch_x = torch.from_numpy(train_x[
                start_idx:start_idx + n_batch]).type(torch.FloatTensor).cuda()
            batch_y = torch.from_numpy(
                train_y[start_idx:start_idx + n_batch]).cuda()
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        n_correct = 0
        for start_idx in tqdm(range(0, test_x.shape[0], n_batch)):
            batch_x = torch.from_numpy(test_x[
                start_idx:start_idx + n_batch]).type(torch.FloatTensor).cuda()
            batch_y = torch.from_numpy(
                test_y[start_idx:start_idx + n_batch]).cuda()
            outputs = model(batch_x)
            _, batch_y_hat = torch.max(outputs.data, 1)
            n_correct += (batch_y_hat == batch_y).sum().item()

        test_acc = n_correct / n_total
        if max_test_acc < test_acc:
            max_test_acc = test_acc
            max_epoch_idx = epoch_idx

        print(epoch_idx, '%.6f' % test_acc, max_epoch_idx,
              '%.6f' % max_test_acc)

    return max_test_acc


def _simulate(args):
    traces = Dataset(args.dataset_idx).traces

    if args.model == 'mlp':
        from models.mlp import MultiLayerPerceptron
        model = MultiLayerPerceptron(args.n_history, 1024, 2)
        model.cuda()
    elif args.model == 'lstm':
        from models.rnn import LSTM
        model = LSTM()
        model.cuda()
    else:
        raise NotImplementedError()

    accuracy = __get_prediction_accuracy(model, traces, args)
    print('>> accuracy     :', '%.8f' % accuracy)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='neural-branch-predictor')
    parser.add_argument(
        '--dataset_idx', type=int, default=1, help='dataset_idx')
    parser.add_argument(
        '--model',
        type=str,
        # default="mlp",
        default="lstm",
        choices=["mlp", "lstm"],
        help='model')
    parser.add_argument(
        '--n_history',
        type=int,
        default=32,
        help='history length for perceptron model. Ignore in counter model.')
    args = parser.parse_args()
    _simulate(args)


if __name__ == '__main__':
    main()
