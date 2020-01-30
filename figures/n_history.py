import re

import matplotlib.pyplot as plt
import numpy as np


def _get_data_from_filename(i, filename):
    f = open('figures/n_histories/{:02d}_{}.dat'.format(i, filename), 'r')
    data = []
    for line in f:
        line = line.strip()
        cols = re.split('\s+', line)
        data.append([float(c) for c in cols])
    return np.array(data)


def main():
    fig, axs = plt.subplots(4, 2, sharey=False, tight_layout=True)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    color = "#757575"
    # color = "#000000"

    filenames = [
        "bzip",
        "gcc",
        "gzip",
        "mcf",
        "parser",
        "twolf",
        "vortex",
        "vpr",
    ]
    for i, filename in enumerate(filenames):
        data = _get_data_from_filename(i, filename)

        ax = axs[i // 2][i % 2]
        ax.set_title(filename)
        ax.plot(data[:, 0], data[:, 1], label='Perceptron', color="#FF4841")
        ax.plot(data[:, 0], data[:, 2], label='Winnow', color="#0099FE")
        ax.set_xscale("log")
        # ax.set_xlim([0, 16384])
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('# of Histories')
        ax.legend(loc='lower left')

    fig.savefig('figures/n_histories/graph.png', format='png')
    fig.savefig('figures/n_histories/graph.pdf', format='pdf')


if __name__ == '__main__':
    main()
