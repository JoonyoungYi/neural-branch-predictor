import os
import re


class Dataset:
    def __init__(self, dataset_idx):
        assert dataset_idx >= 0

        filename = ["00-gcc.out", "01-mcf.out"][dataset_idx]

        traces = []
        with open(os.path.join('datasets', filename), 'r') as f:
            for line in f:
                tokens = re.split('\s+', line.strip())
                assert len(tokens) == 3
                register = tokens[1]
                taken = int(tokens[2])
                assert taken == 0 or taken == 1
                traces.append([register, taken])
        self.traces = traces
