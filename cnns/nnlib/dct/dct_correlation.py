import numpy as np


def dct2(x):
    """
    :param x: 1D input signal in the time domain.
    :return: DCT(x) - 1D output signal in the frequency DCT domain.
    """
    P = x.shape[-1]
    X = np.zeros(P, dtype=float)
    for k in range(P):
        out = 0
        if k == 0:
            kk = 1. / np.sqrt(2.)
        else:
            kk = 1.
        for n in range(P):
            out += kk * x[n] * np.cos(np.pi * (n + .5) * k / P)
        out *= np.sqrt(2. / P)
        X[k] = out
    return X


def dct1(Y):
    N = Y.shape[-1]
    y = np.zeros(N, dtype=float)
    for n in range(N):
        out = 0
        for k in range(N):
            if k == 0:
                kk = 1. / 2.
            else:
                kk = 1.
            out += kk * Y[k] * np.cos(np.pi * n * k / N)
        y[n] = out
    return y


def correlate(x, h):
    """
    Cross-correlation of x and y via the DCT transform.

    :param x: input signal x in the time domain
    :param h: input filter y in the time domain
    :return: the output of the correlation
    """
    N = len(x)
    L = len(h)
    M = N + L - 1
    P1 = max((L - 3), 0) // 2 + 1
    P2 = max(P1 + 1, (N - 3) // 2 + 1)
    # P2 = (N - 3) // 2
    P = max(P2 + 1, 3 * M // 2 + 1)

    x = np.pad(array=x, pad_width=(P1, P - P1 - N), mode='constant')
    h = np.flip(h, axis=0)
    h = np.pad(array=h, pad_width=(P2, P - P2 - L), mode='constant')
    x = dct2(x)
    h = dct2(h)
    y = x * h
    y = dct1(y)[P1 + P2:P1 + P2 + M]
    return y


if __name__ == "__main__":
    x = np.array([1.0, -2.0, 3.0, -4.0, 2.0, -8.0, -1.0, 5.0], dtype=float)
    # x = np.random.randn(21)
    print("x: ", x)
    h = np.array([1.0, -4.0, 2.0], dtype=float)
    h_len = h.shape[-1]
    pad = (h_len - 1) // 2
    x_pad = np.pad(x, [pad, pad], mode="constant")
    expect = np.correlate(x_pad, h, mode='valid')
    print("expect: ", expect)
    result = correlate(x, h)
    print("result: ", result)
    assert np.testing.assert_allclose(actual=result, desired=expect)
