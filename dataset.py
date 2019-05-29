import os
import re


class Dataset:
    def __init__(self, dataset_idx):
        assert dataset_idx >= 0

        # filename = ["00-gcc.out", "01-mcf.out"][dataset_idx]
        filename = [
            "bzip2_result.out",
            "gcc_result.out",
            "gzip_result.out",
            "mcf_result.out",
            "parser_result.out",
            "twolf_result.out",
            "vortex_result.out",
            "vpr_result.out",
        ][dataset_idx]

        traces = []
        with open(os.path.join('datasets', filename), 'r') as f:
            for line in f:
                tokens = re.split('\s+', line.strip())
                if len(tokens) != 2:
                    print(tokens)
                    raise Exception("len(tokens) is not 2.")
                register = tokens[0]
                taken = int(tokens[1])
                assert taken == 0 or taken == 1
                traces.append([register, taken])
        self.traces = traces
