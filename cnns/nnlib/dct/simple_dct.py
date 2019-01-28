import numpy as np
import math

class DCT:

    def __init__(self):
        pass

    def dct2(self, x):
        N = x.shape[-1]
        X = np.zeros(N, dtype=float)
        for k in range(N):
            out = 0
            for n in range(N):
                out += x[n] * np.cos(np.pi * (n + .5) * k / N)
            if k == 0:
                out *= np.sqrt(1. / N)
            else:
                out *= np.sqrt(2. / N)
            X[k] = out
        return X

    def dct1(self, Y):
        N = Y.shape[-1]
        y = np.zeros(N, dtype=float)
        for n in range(N):
            out = 0
            for k in range(N):
                out += Y[k] * np.cos(np.pi * n * k / N)
            if k == 0:
                out *= 1./2
            y[n] = out
        return y

    def correlate(self, x, y, use_next_power2=False):
        """
        Correlate 1D input signals via the DCT transformation.

        :param x: input signal in the time domain.
        :param y: input filter in the time domain.
        :return: the correlation z between x and y
        """
        N = len(x)
        L = len(y)
        M = N + L - 1
        P1 = max((L - 3), 0) // 2 + 1
        P2 = max(P1+1, (N - 3) // 2 + 1)
        P = max(P2 + 1, 3 * M // 2 + 1)

        if use_next_power2:
            P = int(2 ** np.ceil(np.log2(P)))

        x = np.pad(array=x, pad_width=(P1, P - P1 - N), mode='constant')
        # y = np.flip(y)
        y = np.pad(array=y, pad_width=(P2, P - P2 - L), mode='constant')
        x = self.dct2(x)
        y = self.dct2(y)
        z = x * y
        z = self.dct1(z)
        return z


    def idct(self, x):
        N = len(x)
        X = np.zeros(N, dtype=float)
        for k in range(N):
            out = np.sqrt(.5) * x[0]
            for n in range(1, N):
                out += x[n] * np.cos(np.pi * n * (k + .5) / N)
            X[k] = out * np.sqrt(2. / N)
        return X

if __name__ == "__main__":
    dct = DCT()
    # x = np.array([1.0, 2.0, 3.0, 4.0])
    # y = np.array([-1.0, 3.0])
    x = np.arange(0, 21, dtype=float)
    y = np.array([1.0, 2.0, 3.0])
    expect = np.correlate(x, y)
    print("expect: ", expect)
    result = dct.correlate(x, y)
    print("result: ", result)
    assert np.testing.assert_allclose(actual=result, desired=expect)
