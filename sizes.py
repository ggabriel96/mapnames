import json

import numpy as np


def stats(arr):
    return {
        'min': np.min(arr),
        'max': np.max(arr),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'q1': np.percentile(arr, 25),
        'q2': np.percentile(arr, 50),
        'q3': np.percentile(arr, 75)
    }


for i in range(24):
    with open(f'case{i}.json') as f:
        js = json.load(f)
        all_strs = list(js.keys()) + list(js.values())
        arr = [len(s) for s in all_strs]
        sts = stats(np.array(arr))
        print('{:2} & {:8.1e}'.format(i + 1, len(js)), end='')
        for v in sts.values():
            if isinstance(v, np.int64):
                print(' & {:8}'.format(v), end='')
            else:
                print(' & {:8.04}'.format(v), end='')
        print()
