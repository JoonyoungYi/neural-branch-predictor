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

# tf-4: python3 run.py --model=winnow --n_history=8192
# tf-6: python3 runner.py --dataset_idx=5 --n_history=32768 --model=winnow && python3 runner.py --dataset_idx=7 --n_history=32768 --model=winnow
# tf-1: python3 run.py --model=winnow --n_history=16384
# tf-3: python3 runner.py --dataset_idx=2 --n_history=32768 --model=winnow && python3 runner.py --dataset_idx=4 --n_history=32768 --model=winnow

# tf-7: python3 run.py --model=perceptron --n_history=8192
# tf-0: python3 run.py --model=perceptron --n_history=16384
# tf-2: python3 runner.py --dataset_idx=5 --n_history=32768 --model=perceptron && python3 runner.py --dataset_idx=7 --n_history=32768 --model=perceptron
# tf-5: python3 runner.py --dataset_idx=2 --n_history=32768 --model=perceptron && python3 runner.py --dataset_idx=4 --n_history=32768 --model=perceptron
