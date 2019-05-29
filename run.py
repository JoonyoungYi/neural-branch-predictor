from dataset import Dataset
from configs import TEST_SPLIT


def __get_prediction_accuracy(model, traces):
    train_traces = traces[:int(len(traces) * (1 - TEST_SPLIT))]
    test_traces = traces[int(len(traces) * (1 - TEST_SPLIT)):]

    for register, taken in train_traces:
        model.train(register, taken)

    n_correct = 0
    for register, taken in test_traces:
        taken_hat = model.predict(register)
        if taken_hat == taken:
            n_correct += 1
        model.train(register, taken)
    return n_correct / len(test_traces)


def _simulate(args):
    traces = Dataset(args.dataset_idx).traces

    if args.model == 'saturating-counter':
        from models.saturating_counter import SaturatingCounter
        model = SaturatingCounter()
    elif args.model == 'perceptron':
        from models.perceptron import Perceptron
        model = Perceptron(N=args.n_history)
    else:
        raise NotImplementedError()

    accuracy = __get_prediction_accuracy(model, traces)
    print('%.5f' % accuracy)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='neural-branch-predictor')
    parser.add_argument(
        '--dataset_idx', type=int, default=0, help='dataset_idx')
    parser.add_argument(
        '--model',
        type=str,
        default="saturating-counter",
        # default="perceptron",
        choices=["saturating-counter", "perceptron"],
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
