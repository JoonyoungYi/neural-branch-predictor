import time

from dataset import Dataset
from configs import TEST_SPLIT


def __get_prediction_accuracy(model, traces):
    train_traces = traces[:int(len(traces) * (1 - TEST_SPLIT))]
    test_traces = traces[int(len(traces) * (1 - TEST_SPLIT)):]

    start_time = time.time()
    for register, taken in train_traces:
        model.train(register, taken)
    print(">> training(sec):", time.time() - start_time)

    n_correct = 0
    start_time = time.time()
    for register, taken in test_traces:
        taken_hat = model.predict(register)
        assert taken_hat == 0 or taken_hat == 1
        if taken_hat == taken:
            n_correct += 1
        model.train(register, taken)
    print(">> testing (sec):", time.time() - start_time)

    return n_correct / len(test_traces)


def _simulate(args):
    traces = Dataset(args.dataset_idx).traces

    if args.model == 'saturating-counter':
        from models.saturating_counter import SaturatingCounterPredictor
        model = SaturatingCounterPredictor()
    elif args.model == 'perceptron':
        from models.perceptron import PerceptronPredictor
        model = PerceptronPredictor(N=args.n_history)
    elif args.model == 'constant':
        from models.constant import ConstantPredictor
        model = ConstantPredictor(1)
    else:
        raise NotImplementedError()

    accuracy = __get_prediction_accuracy(model, traces)
    print('>> accuracy     :', '%.8f' % accuracy)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='neural-branch-predictor')
    parser.add_argument(
        '--dataset_idx', type=int, default=1, help='dataset_idx')
    parser.add_argument(
        '--model',
        type=str,
        # default="saturating-counter",
        default="perceptron",
        # default="constant",
        choices=["saturating-counter", "perceptron", "constant"],
        help='model')
    parser.add_argument(
        '--n_history',
        type=int,
        default=8,
        help='history length for perceptron model. Ignore in counter model.')
    args = parser.parse_args()
    _simulate(args)


if __name__ == '__main__':
    main()
