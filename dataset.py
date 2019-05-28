import os
import re


class Dataset:
    def __init__(self, dataset_idx):
        assert dataset_idx

        filename = ["01-gcc.out", "02-mcf.out"][dataset_idx]

        traces = []
        with open(os.path.join('datasets', filename), 'r') as f:
            for line in f:
                tokens = re.split('\s+', line.strip())
                assert len(tokens) == 3
                register = tokens[1]
                taken = tokens[2] == '1'
                traces.append([register, taken])
        self.traces = traces
