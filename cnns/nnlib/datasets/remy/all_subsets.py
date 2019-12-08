# all subsets

import itertools
import numpy as np


def findsubsets(s):
    n = len(s) + 1
    subsets = []
    for i in range(1, n):
        subset = list(itertools.combinations(s, i))
        for x in subset:
            subsets.append(x)
    return subsets


for n in range(30):
    print(n)
    findsubsets(np.arange(n))
