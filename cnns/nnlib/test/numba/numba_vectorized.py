import numpy as np
from timeit import default_timer as timer

"""
Timing: 7.19 sec

c first 5 elements:  [0.69906056 0.959825   0.8178963  0.9870413  0.7213005 ]
c last 5 elements:  [0.83658946 0.71512336 0.6924522  0.694939   0.8647377 ]
duration:  7.187152965925634
"""


def pow(a, b):
    return a ** b


def main():
    np.random.seed(31)
    vec_size = 100000000
    # vec_size = 100

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
