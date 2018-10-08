import numpy as np
from timeit import default_timer as timer

"""
Timing: 38.54130343068391 sec
"""


def pow(a, b, c):
    for i in range(a.size):
        c[i] = a[i] ** b[i]


def main():
    np.random.seed(31)
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    pow(a, b, c)
    duration = timer() - start

    print("c first 5 elements: ", c[:5])
    print("c last 5 elements: ", c[-5:])

    print("duration: ", duration)


if __name__ == '__main__':
    main()
