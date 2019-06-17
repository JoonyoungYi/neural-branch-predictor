import time

try:
    from tqdm import tqdm
except:
    print('There is no tqdm...')
    tqdm = lambda x: x

from dataset import Dataset


def __get_prediction_accuracy(model, traces):

    start_time = time.time()
    n_correct = 0
    results = []
    for register, taken in tqdm(traces):
        taken_hat = model.predict(register)
        assert taken_hat == 0 or taken_hat == 1

        if taken_hat == taken:
            n_correct += 1
        results.append(1 if taken_hat == taken else 0)

        model.train(register, taken)

    # print(">> running (sec):", time.time() - start_time)
    # print(results)
    # input('')
    return n_correct / len(traces), time.time() - start_time


def _simulate(args):
    traces = Dataset(args.dataset_idx).traces

    if args.model == 'saturating-counter':
        from models.saturating_counter import SaturatingCounterPredictor
        model = SaturatingCounterPredictor()
    elif args.model == 'perceptron':
        from models.perceptron import PerceptronPredictor
        model = PerceptronPredictor(N=args.n_history)
    elif args.model == 'winnow':
        from models.winnow import WinnowPredictor
        model = WinnowPredictor(N=args.n_history)
    elif args.model == 'constant':
        from models.constant import ConstantPredictor
        model = ConstantPredictor(1)
    else:
        raise NotImplementedError()

    accuracy, time = __get_prediction_accuracy(model, traces)
    print('>> accuracy     :', '%.8f' % accuracy)

    f = open('results/{}.tsv'.format(args.model), 'a')
    results = []
    results.append(args.dataset_idx)
    results.append(args.n_history)
    results.append(accuracy)
    results.append(time)
    f.write('\t'.join([str(r) for r in results]) + '\n')
    f.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='neural-branch-predictor')
    parser.add_argument(
        '--dataset_idx', type=int, default=1, help='dataset_idx')
    parser.add_argument(
        '--model',
        type=str,
        # default="saturating-counter",
        # default="perceptron",
        default="winnow",
        # default="constant",
        choices=["saturating-counter", "perceptron", "constant", "winnow"],
        help='model')
    parser.add_argument(
        '--n_history',
        type=int,
        default=1,
        help='history length for perceptron model. Ignore in counter model.')
    args = parser.parse_args()
    _simulate(args)


if __name__ == '__main__':
    main()
