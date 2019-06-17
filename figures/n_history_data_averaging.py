import re

import matplotlib.pyplot as plt
import numpy as np


def _get_data():
    f = open('figures/n_history_data_averaging/data.dat', 'r')
    data = []
    for line in f:
        line = line.strip()
        cols = re.split('\s+', line)
        data.append([float(c) for c in cols])
    return np.array(data)


def main():
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True)
    fig.set_figwidth(6)
    fig.set_figheight(4)

    data = _get_data()
    ax = axs
    ax.set_title('Accuracy (Perceptron vs Winnow)')
    ax.plot(data[:, 0], data[:, 1], label='Perceptron', color="#FF4841")
    l = 1.645
    plt.fill_between(
        data[:, 0],
        data[:, 1] - l * data[:, 2],
        data[:, 1] + l * data[:, 2],
        color="#FF4841",
        alpha=0.1)
    ax.plot(data[:, 0], data[:, 3], label='Winnow', color="#0099FE")
    plt.fill_between(
        data[:, 0],
        data[:, 3] - l * data[:, 4],
        data[:, 3] + l * data[:, 4],
        color="#0099FE",
        alpha=0.1)
    plt.xlim([1, 16384])
    ax.set_xscale("log")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# of Histories')
    ax.legend(loc='lower left')

    fig.savefig('figures/n_history_data_averaging/graph.png', format='png')
    fig.savefig('figures/n_history_data_averaging/graph.pdf', format='pdf')


if __name__ == '__main__':
    main()
