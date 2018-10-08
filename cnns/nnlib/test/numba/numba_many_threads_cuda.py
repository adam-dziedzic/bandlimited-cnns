import numpy as np
from timeit import default_timer as timer
from numba import vectorize

"""
c first 5 elements:  [0.6990605  0.959825   0.8178963  0.98704123 0.7213005 ]
c last 5 elements:  [0.8365894  0.7151233  0.6924522  0.69493896 0.86473775]
duration:  0.7203962397761643 sec
"""

@vectorize(['float32(float32, float32)'], target='cuda')
def pow(a, b):
    return a ** b


def main():
    np.random.seed(31)
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = pow(a, b)
    duration = timer() - start

    print("c first 5 elements: ", c[:5])
    print("c last 5 elements: ", c[-5:])

    print("duration: ", duration)


if __name__ == '__main__':
    main()