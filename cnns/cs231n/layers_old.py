from builtins import range

import matplotlib.pyplot as plt
import numpy as np
import pyfftw


def conv_forward_fftw_1D(x, w, b, conv_param, preserve_energy_rate=1.0):
    """
    Forward pass of 1D convolution.

    The input consists of N data points with each data point representing a time-series of length W.

    We also have the notion of channels in the 1-D convolution. We want to use more than a single filter even for the
    input time-series, so the output is a the batch with the same size but the number of output channels is equal to the
    number of input filters.

    :param x: Input data of shape (N, C, W)
    :param w: Filter weights of shape (F, C, WW)
    :param b: biases, of shape (F,)
    :param conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    # Grab conv parameters
    # print("conv_param: ", conv_param)
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    N, C, W = x.shape
    F, C, WW = w.shape

    xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)

    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))

    # Calculate output spatial dimensions.
    out_W = np.int(((W + 2 * pad - WW) / stride) + 1)

    # Initialise the output.
    out = np.zeros([N, F, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                # xfft = np.fft.fft(padded_x[nn, cc], fftsize)
                xfft = pyfftw.interfaces.numpy_fft.fft(padded_x[nn, cc], fftsize)
                # print("first xfft: ", xfft)
                # xfft = xfft[:len(xfft) // 2]
                # squared_abs = np.abs(xfft) ** 2
                # full_energy = np.sum(squared_abs)
                # current_energy = 0.0
                # preserve_energy = full_energy * preserve_energy_rate
                # index = 0
                # while current_energy < preserve_energy and index < len(squared_abs):
                #     current_energy += squared_abs[index]
                #     index += 1
                # print("index: ", index)
                # xfft = xfft[:index]
                # print("xfft: ", xfft)
                # xfft = xfft[:xfft.shape[0] // 2, :xfft.shape[1] // 2]
                # print("xfft shape: ", xfft.shape)
                filters = w[ff, cc]
                # print("filters: ", filters)
                # print("last shape of xfft: ", xfft.shape[-1])
                # The convolution theorem takes the duration of the response to be the same as the period of the data.
                # filterfft = np.fft.fft(filters, xfft.shape[-1])
                filterfft = pyfftw.interfaces.numpy_fft.fft(filters, xfft.shape[-1])
                # filterfft = np.fft.fft(filters, xfft.shape[-1]*2)
                # filterfft = filterfft[:filterfft.shape[0] // 2, :filterfft.shape[1] // 2]
                # filterfft = filterfft[:filterfft.shape[-1] // 2]
                # print("filterfft: ", filterfft)
                filterfft = np.conj(filterfft)
                outfft = xfft * filterfft
                # outfft = np.concatenate(outfft, reversed(outfft))
                # take the inverse of the output from the frequency domain and return the modules of the complex numbers
                # outifft = np.fft.ifft(outfft)
                outifft = pyfftw.interfaces.numpy_fft.ifft(outfft)
                # out[nn, ff] += np.abs(np.fft.ifft2(xfft * filterfft, (out_H, out_W)))
                # outdouble = np.array(outifft, np.double)
                out_real = np.real(outifft)
                # out_real = np.abs(outifft)
                if len(out_real) < out_W:
                    out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                sum_out += out_real[:out_W]
            # crop the output to the expected shape
            # print("shape of expected resuls: ", out[nn, ff].shape)
            # print("shape of sum_out: ", sum_out.shape)
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_fft_1D_compress(x, w, b, conv_param, preserve_energy_rate=1.0):
    """
    Forward pass of 1D convolution.

    The input consists of N data points with each data point representing a time-series of length W.

    We also have the notion of channels in the 1-D convolution. We want to use more than a single filter even for the
    input time-series, so the output is a batch with the same size but the number of output channels is equal to the
    number of input filters.

    :param x: Input data of shape (N, C, W)
    :param w: Filter weights of shape (F, C, WW)
    :param b: biases, of shape (F,)
    :param conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    :param preserve_energy_rate - percentage of energy preserved for the signal in the frequency domain by preserving
    only a specific number (found in iterative way) of the first Fourier coefficients.
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    # Grab conv parameters
    # print("conv_param: ", conv_param)
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    N, C, W = x.shape
    F, C, WW = w.shape

    xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)

    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))

    # Calculate output spatial dimensions.
    out_W = np.int(((W + 2 * pad - WW) / stride) + 1)

    # Initialise the output.
    out = np.zeros([N, F, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = np.fft.fft(padded_x[nn, cc], fftsize)
                # print("first xfft: ", xfft)
                # print("preserve_energy_rate: ", preserve_energy_rate)
                # fig = plt.figure()
                # ax1 = fig.add_subplot(111)
                # x = [x for x in range(len(xfft))]
                # ax1.xcorr(x, xfft, usevlines=True, maxlags=50, normed=True, lw=2)
                # ax1.grid(True)
                # ax1.axhline(0, color='black', lw=2)
                # plt.show()
                xfft_middle = len(xfft) // 2
                xfft_power = xfft[0:xfft_middle + 1]
                squared_abs = np.abs(xfft_power) ** 2
                full_energy = np.sum(squared_abs)
                # we always include the first and the middle coefficients
                current_energy = squared_abs[0] + squared_abs[xfft_middle]
                # print("full energy: ", full_energy)
                preserve_energy = full_energy * preserve_energy_rate
                index = 1
                while current_energy < preserve_energy and index < len(squared_abs) - 1:
                    current_energy += squared_abs[index]
                    index += 1
                # print("index: ", index)
                # print("shape of np.array(xfft[:index]): ", np.array(xfft[:index].shape))
                # print("shape of np.array(xfft[-index + 1:]): ", np.array(xfft[-index + 1:]).shape)
                xfft = np.concatenate((np.array(xfft[0:index]), np.array(xfft[xfft_middle:xfft_middle + 1]),
                                       np.array(xfft[-index + 1:])))
                # flen = len(xfft) // 2
                # first_half = xfft[1:flen]
                # print("xfft first half: ", first_half)
                # second_half = xfft[flen + 1:]
                # second_half = np.flip(np.conj(second_half), axis=0)
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(xfft)), np.abs(xfft))
                # plt.title("dataset: " + "50words" + ", preserved energy: " + str(preserve_energy_rate))
                # plt.xlabel('index')
                # plt.ylabel('Absolute value')
                # plt.show()
                # print("are halves close: ", np.allclose(first_half, second_half))
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(xfft)), np.abs(xfft))
                # plt.xlabel('index')
                # plt.ylabel('Absolute value')
                # plt.show()
                # print("xfft: ", xfft)
                # xfft = xfft[:xfft.shape[0] // 2, :xfft.shape[1] // 2]
                # print("xfft shape: ", xfft.shape)
                filters = w[ff, cc]
                # print("filters: ", filters)
                # print("last shape of xfft: ", xfft.shape[-1])
                # The convolution theorem takes the duration of the response to be the same as the period of the data.
                filterfft = np.fft.fft(filters, xfft.shape[-1])
                # print("filterfft: ", filterfft)
                flen = len(filterfft) // 2
                # print("filter middle: ", filterfft[flen])
                first_half = filterfft[1:flen]
                # print("filter first half: ", first_half)
                second_half = filterfft[flen + 1:]
                second_half = np.flip(np.conj(second_half), axis=0)
                # print("filter second half: ", second_half)
                # print("first half length: ", len(first_half))
                # print("second half length: ", len(second_half))
                # print("first coefficient from the first half: ", first_half[0])
                # print("first coefficient from the second half: ", second_half[0])
                # print("are first coefficients equal: ", first_half[0] == second_half[0])
                # print("are halves equal: ", filterfft[1:flen] == second_half)
                # print("are halves close: ", np.allclose(first_half, second_half))
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(filterfft)), np.abs(filterfft))
                # plt.title("filter")
                # plt.xlabel('index')
                # plt.ylabel('Absolute value')
                # plt.show()
                # filterfft = np.fft.fft(filters, xfft.shape[-1]*2)
                # filterfft = filterfft[:filterfft.shape[0] // 2, :filterfft.shape[1] // 2]
                # filterfft = filterfft[:filterfft.shape[-1] // 2]
                # print("filterfft: ", filterfft)
                filterfft = np.conj(filterfft)
                outfft = xfft * filterfft
                # outfft = np.concatenate(outfft, reversed(outfft))
                # take the inverse of the output from the frequency domain and return the modules of the complex numbers
                outifft = np.fft.ifft(outfft)
                # out[nn, ff] += np.abs(np.fft.ifft2(xfft * filterfft, (out_H, out_W)))
                # outdouble = np.array(outifft, np.double)
                out_real = np.real(outifft)
                # out_real = np.abs(outifft)
                if len(out_real) < out_W:
                    out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                sum_out += out_real[:out_W]
            # crop the output to the expected shape
            # print("shape of expected resuls: ", out[nn, ff].shape)
            # print("shape of sum_out: ", sum_out.shape)
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache

def cross_correlation_fft_1D(x, w, pad):
    N, C, W=x.shape
    F, C, WW = w.shape

    xw_size = W + WW - 1
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
    out_W = W + 2 * pad - WW + 1
    xfft = np.fft.fft(padded_x, fftsize)
    filterfft = np.fft.fft(w, xfft.shape[-1])
    filterfft = np.conj(filterfft)
    outfft = xfft * filterfft
    outifft = np.fft.ifft(outfft)
    out_real = np.real(outifft)