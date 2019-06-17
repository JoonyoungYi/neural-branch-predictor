import os

import argparse

parser = argparse.ArgumentParser(description='neural-branch-predictor')
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

for dataset_idx in range(8):
    os.system("python3 runner.py --model={} --n_history={} --dataset_idx={}".
              format(args.model, args.n_history, dataset_idx))
