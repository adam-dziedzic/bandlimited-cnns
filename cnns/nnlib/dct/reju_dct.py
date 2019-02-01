"""
Circular convolution using Discrete sine and cosine transforms.

Reference: V. G. Reju, S. N. Koh and I. Y. Soon, Convolution Using Discrete
Sine and Cosine Transforms,
IEEE Signal Processing Letters, VOL. 14, NO. 7, JULY 2007, pp.445-448.
"""

import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft, rfft, irfft

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


font = {'size': 20}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))


def DCT1e(x):
    """
    This can act as a forward or backward transform. We use it simply as a
    transform. This is a very similar definition to wikipedia but with
    scaling 2.
    https://www.researchgate.net/publication/3343693_Convolution_Using_Discrete_Sine_and_Cosine_Transforms
    DCT1 of the input signal x.
    :param x: (N+1) point input signal.
    :return: transformed x.
    """
    N = x.shape[-1]
    X = np.zeros(N, dtype=float)
    for k in range(N):
        out = 0.0
        for n in range(N):
            if n == 0 or n == (N - 1):
                kk = 1.0 / 2
            else:
                kk = 1.0
            out += kk * x[n] * np.cos(np.pi * k * n / (N - 1))
        X[k] = 2 * out
    return X


def DCT2e(x):
    """
    This can act as a forward or backward transform. We use it simply as a
    transform. This is a very similar definition to wikipedia but with
    scaling 2.
    https://www.researchgate.net/publication/3343693_Convolution_Using_Discrete_Sine_and_Cosine_Transforms
    DCT2 of the input signal x.
    :param x: (N) point input signal.
    :return: transformed x.
    """
    N = x.shape[-1]
    X = np.zeros(N + 1, dtype=float)
    for k in range(N):
        out = 0.0
        for n in range(N):
            out += x[n] * np.cos(np.pi * k * (n + 0.5) / N)
        X[k] = 2 * out
    X[N] = 0.0
    return X


def DST1e(x):
    """
    This can act as a forward or backward transform. We use it simply as a
    transform. This is a very similar definition to wikipedia but with
    scaling 2.
    https://www.researchgate.net/publication/3343693_Convolution_Using_Discrete_Sine_and_Cosine_Transforms
    DST1 of the input signal x.
    :param x: (N) point input signal.
    :return: transformed x.
    """
    N = x.shape[-1]
    X = np.zeros(N, dtype=float)
    for k in range(1, N + 1):
        out = 0.0
        for n in range(1, N + 1):
            out += x[n - 1] * np.sin(np.pi * k * n / (N + 1))
        X[k - 1] = 2 * out
    return X


def DST2e(x):
    """
    This can act as a forward or backward transform. We use it simply as a
    transform. This is a very similar definition to wikipedia but with
    scaling 2.
    https://www.researchgate.net/publication/3343693_Convolution_Using_Discrete_Sine_and_Cosine_Transforms
    DST2 of the input signal x.
    :param x: (N) point input signal.
    :return: transformed x.
    """
    N = x.shape[-1]
    X = np.zeros(N, dtype=float)
    for k in range(1, N + 1):
        out = 0.0
        for n in range(N):
            out += x[n] * np.sin(np.pi * k * (n + 0.5) / N)
        X[k - 1] = 2 * out
    return X


def dC2e(x):
    """
    Apply DCT to x and make the signal even - symmetric.

    function dsC2e=dC2e(s)
    N=length(s);
    fprintf("N: %i\n", N)
    sC2e=DCT2e(s); %n=0:N-1; k=0:N-1
    fprintf("DCT2e");
    display(sC2e);
    if rem(N,2)==0  %For even N
        temp=sC2e(1:2:end-1);
        dsC2e=[temp; 0; -1*flipud(temp(2:end)); 0];

    else    %For odd N
        temp=sC2e(1:2:end);
        dsC2e=[temp; -1*flipud(temp(2:end)); 0];

    end
    """
    N = x.shape[-1]
    sC2e = DCT2e(x)
    if N % 2 == 0:  # N is even
        temp = sC2e[0:N - 1:2]
        dsC2e = np.concatenate(
            [temp, [0.0], -1.0 * np.flip(temp[1:], axis=0), [0.0]])
    else:  # N is odd
        temp = sC2e[0::2]  # decimation: select every other element
        dsC2e = np.concatenate([temp, -1.0 * np.flip(temp[1:], axis=0), [0.0]])
    return dsC2e


def dS2e(x):
    """
    function dsS2e=dS2e(s)
    N=length(s);
    sS2e=DST2e(s); %n=0:N-1, k=1:N
    if rem(N,2)==0  %For even N
        temp=sS2e(2:2:end-2);
        dsS2e=[0; temp; sS2e(end); flipud(temp); 0];
    else    %For odd N
        temp=sS2e(2:2:end-1);
        dsS2e=[0; temp; flipud(temp); 0];
    end
    :param x:
    :return:
    """
    N = x.shape[-1]
    sS2e = DST2e(x)
    if N % 2 == 0:  # N is even
        temp = sS2e[1:N - 1:2]
        dsS2e = np.concatenate(
            [[0.0], temp, [sS2e[-1]], np.flip(temp, axis=0), [0.0]])
    else:  # N is odd
        temp = sS2e[1:N - 1:2]  # decimation: select every other element
        dsS2e = np.concatenate([[0.0], temp, np.flip(temp, axis=0), [0.0]])
    return dsS2e


def rfft_convolution(x, y):
    # Convolution by DFT method (Discrete Fourier Transform).
    zdft = np.real(irfft(rfft(x) * rfft(y)))
    return zdft


def fft_convolution(x, y):
    # Convolution by DFT method (Discrete Fourier Transform).
    zdft = np.real(ifft(fft(x) * fft(y)))
    return zdft


def dct_convolution(s, h):
    N = s.shape[-1]  # the length of the signals
    M = h.shape[-1]
    assert N == M

    # Calculation of T1
    dC2es = dC2e(s)
    dS2es = dS2e(s)
    dC2eh = dC2e(h)
    dS2eh = dS2e(h)
    # T1 = dC2e(s) * dC2e(h) - dS2e(s) * dS2e(h)
    T1 = dC2es * dC2eh - dS2es * dS2eh
    # calculation of T2
    # T2 = dS2e(s) * dC2e(h) + dC2e(s) * dS2e(h)
    T2 = dS2es * dC2eh + dC2es * dS2eh
    # after calculating T2, remove zero-th and N-th values from T2 because for
    # S1e the range is from n=1:N-1 but T2 calculated is for n=0:N
    T2 = T2[1:-1]
    # Multiply the first and last element of T1 by k_{k}=2
    T1[0] = T1[0] * 2
    # T1[0] will have some non-zeros value, which corresponds to
    # the DC component. So if we don's multiply , there will be a DC shift in
    # the final convolution output. So in BSS, the scaling on this component
    # may not cause any problem.
    # If the signals to be convolved are of zero mean, there won't be any dc
    # and hence T1[0] will be zero.

    # T1[-1] is always zero, so there is no effect in multiplying but for the
    # compatability with the std equation we can multiply.
    T1[-1] = T1[-1] * 2

    # Calculation of dT1c1e=dC1e{T1}
    # dT1c1e=dC1e*T1
    T1C1e = DCT1e(T1)
    if N % 2 == 0:  # For even N
        temp = T1C1e[0:-2:2]
        dT1C1e = np.concatenate([temp, [T1C1e[-1]], np.flip(temp, axis=0)])
    else:
        temp = T1C1e[0:-1:2]
        dT1C1e = np.concatenate([temp, np.flip(temp, axis=0)])

    # Calculation of dT2s1e=dS1e{T2}
    # dT2s1e=dS1e*T2
    T2S1e = DST1e(T2)
    if N % 2 == 0:
        temp = T2S1e[1:-1:2]
        dT2S1e = np.concatenate([temp, [0.0], -1 * np.flip(temp, axis=0)])
    else:
        temp = T2S1e[1::2]
        dT2S1e = np.concatenate([temp, -1 * np.flip(temp, axis=0)])

    y = 1 / (8 * N) * (dT1C1e[1:] + np.concatenate([dT2S1e, [0.0]]))
    return y


def plot_signals(ydct, ydft):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Convolution output using DFT method and proposed DTT method, '
              'to verify the proposed method.')
    plt.plot(ydct, label="DCT", lw=3,
             marker="o", color=get_color(MY_ORANGE))
    plt.plot(ydft, label="DFT", lw=3,
             marker="s", color=get_color(MY_BLUE))
    plt.show()
    fig.savefig(dir_path + "/" + "dft-vs-reju-dct.pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    N = 20  # length of the sequences.
    s = np.random.rand(N, 1)  # first sequence.
    h = np.random.rand(N, 1)  # second sequence.
    out_fft = rfft_convolution(s, h)
    print("out fft: ", out_fft)
    out_dct = dct_convolution(s, h)
    print("out dct: ", out_dct)
