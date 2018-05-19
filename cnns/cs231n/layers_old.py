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


def conv_forward_fft_1D_compress_fraction(x, w, b, conv_param, back_index=0):
    """
    Arbitrarily compress half of the input signal in the frequency domain.
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
    :param fraction - fraction of the signal in the frequency domain, i.e. the first Fourier coefficients that are
    preserved
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    N, C, W = x.shape
    F, C, WW = w.shape

    print("size of x: ", W)
    # xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    # fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    fft_size = 1 << (2 * W - 1).bit_length()
    # fft_size = 2 * W - 1
    print("fft_size: ", fft_size)

    # Zero pad our tensor along the spatial/time domain dimensions.xs
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    # padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
    # padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))

    # Calculate output spatial dimensions.
    out_W = np.int(((W + 2 * pad - WW) / stride) + 1)

    # Initialise the output.
    out = np.zeros([N, F, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                # xfft = np.fft.fft(padded_x[nn, cc], fft_size)
                xfft = np.fft.fft(x[nn, cc], fft_size)
                # print("size of the first xfft: ", xfft.shape)
                # xfft_middle = len(xfft) // 2
                # xfft_middle = len(xfft)
                # xfft_power = xfft[0:xfft_middle+1]
                # squared_abs = np.abs(xfft_power) ** 2
                # full_energy = np.sum(squared_abs)
                # # we always include the first and the middle coefficients
                # current_energy = squared_abs[0] + squared_abs[xfft_middle]
                # # print("full energy: ", full_energy)
                # preserve_energy = full_energy * preserve_energy_rate
                # index = 1
                # while current_energy < preserve_energy and index < len(squared_abs) - 1:
                #     current_energy += squared_abs[index]
                #     index += 1
                # print("index: ", index)
                # print("shape of np.array(xfft[:index]): ", np.array(xfft[:index].shape))
                # print("shape of np.array(xfft[-index + 1:]): ", np.array(xfft[-index + 1:]).shape)
                # compress_size = int(fftsize * fraction)
                # index = (compress_size - 2) // 2  # the two half are the same (with respect to the conjugate operation)
                # index += 1  # we already included the 0-th coefficient (+1)
                # xfft = np.concatenate((np.array(xfft[0:index]), np.array(xfft[xfft_middle:xfft_middle+1]),
                #                       np.array(xfft[-index + 1:])))
                # print("size of compressed xfft: ", xfft.shape)
                # xfft = xfft[0:xfft_middle-back_index]
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
                filterfft = np.fft.fft(filters, fft_size)
                # fmiddle = len(filterfft) // 2
                # fmiddle = len(filterfft)
                # filterfft = filterfft[0:fmiddle-back_index]
                # print("filterfft: ", filterfft)
                # flen = len(filterfft) // 2
                # print("filter middle: ", filterfft[flen])
                # first_half = filterfft[1:flen]
                # print("filter first half: ", first_half)
                # second_half = filterfft[flen + 1:]
                # second_half = np.flip(np.conj(second_half), axis=0)
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
                # outfft_W = np.zeros(max(out_W, 2 * len(outfft)))
                # outfft_W[0:len(outfft)] = np.conj(outfft)
                # outfft_W[-len(outfft):] = np.conj([x for x in reversed(outfft)])
                # outifft = np.fft.ifft(np.conj([x for x in reversed(outfft)]), out_W)
                # fft_combined = np.concatenate((outfft, np.conj(np.flip(outfft, axis=0))))
                # fft_combined = np.concatenate((np.conj(np.flip(outfft, axis=0)), outfft))
                # fft_combined = np.concatenate((np.flip(outfft, axis=0), outfft))
                outifft = np.fft.ifft(outfft)
                # out[nn, ff] += np.abs(np.fft.ifft2(xfft * filterfft, (out_H, out_W)))
                # outdouble = np.array(outifft, np.double)
                # out_size = out_W // 2
                out = np.concatenate((outifft[-(W - 1):], outifft[:W]))
                out_real = np.real(out)
                out_real = out_real[:out_W]
                # print(out_real.shape)
                # out = out[:out_W]
                # out_real = np.concatenate((np.flip(outreal, axis=0), outreal))
                # x_len = W
                # out_real = np.concatenate((out_real[-(x_len - 1):], out_real[:x_len]))
                # out_real = np.abs(outifft)
                # if len(out_real) < out_W:
                #     out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                # out_real *= 2
                import matplotlib.pyplot as plt
                plt.plot(range(0, len(out_real)), out_real)
                plt.title("cross-correlation output compressed")
                plt.xlabel('time')
                plt.ylabel('Amplitude')
                plt.show()
                sum_out += out_real[:out_W]
            # crop the output to the expected shape
            # print("shape of expected results: ", out[nn, ff].shape)
            # print("shape of sum_out: ", sum_out.shape)
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache


def original_ncc_c(x, y):
    from numpy.linalg import norm
    from numpy.fft import fft, ifft
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    y_len = len(y)
    # fft_size = 1 << (2 * x_len - 1).bit_length()
    fft_size = 1 << (x_len + y_len - 1).bit_length()
    # fft_size *= 2
    pad = y_len
    x = (np.pad(x, (pad, 0), 'constant'))
    print("fft_size: ", fft_size)
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    # print("cc: ", cc)
    print("cc size: ", len(cc))
    print("x_len: ", x_len)
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return_value = np.real(cc) / den
    import matplotlib.pyplot as plt
    plt.plot(range(0, len(return_value)), return_value)
    plt.title("cross-correlation output _ncc_c")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()
    return return_value


def cross_corelate_john(x, y, pad):
    x_len = len(x)
    y_len = len(y)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    print("fft_size: ", fft_size)
    # fft_size = 1024
    out_W = x_len + 2 * pad - y_len + 1
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return_value = np.real(cc)
    return_value = return_value[-out_W:]
    import matplotlib.pyplot as plt
    plt.plot(range(0, len(return_value)), return_value)
    plt.title("cross-correlation john")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()
    return return_value


def cross_correlate_adam(x, y, pad):
    x_len = len(x)
    y_len = len(y)
    fft_size = 2 ** np.ceil(np.log2(x_len + y_len - 1)).astype(int)
    padded_x = np.pad(x, (pad, pad), 'constant')
    out_W = x_len + 2 * pad - y_len + 1
    cc = ifft(fft(padded_x, fft_size) * np.conj(fft(y, fft_size)))
    return_value = np.real(cc)
    return_value = return_value[:out_W]
    if len(return_value) < out_W:
        return_value = np.pad(return_value, (0, out_W - len(return_value)), 'constant')
    import matplotlib.pyplot as plt
    plt.plot(range(0, len(return_value)), return_value)
    plt.title("cross-correlation adam")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()
    return return_value


def cross_correlate(x, y):
    from numpy.fft import fft, ifft
    x_len = len(x)
    # fft_size = 1 << (2*x_len-1).bit_length()
    xw_size = len(x) + len(y) - 1
    # The FFT is faster if the input signal is a power of 2.
    fft_size = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    pad = len(y) - 1
    # pad = 0
    padded_x = (np.pad(x, (pad, pad), 'constant'))
    cc = ifft(fft(padded_x, fft_size) * np.conj(fft(y, fft_size)))
    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    cc = np.real(cc)
    # Calculate output spatial/time domain dimensions.
    out_size = len(x) + 2 * pad - len(y) + 1
    cc = cc[:out_size]
    import matplotlib.pyplot as plt
    plt.plot(range(0, len(cc)), cc)
    plt.title("cross-correlation output cross_correlate")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()
    return cc


def cross_correlate_test(x, y):
    from numpy.fft import fft, ifft

    xfft1 = fft(x)
    xfft2 = fft(x, 2 * len(x))
    pad = len(x) // 2
    padded_x = (np.pad(x, (pad, pad), 'constant'))
    xfft3 = fft(padded_x)
    import matplotlib.pyplot as plt
    # plt.plot(range(0, len(xfft1)), np.abs(xfft1), color="red")
    plt.plot(range(0, len(xfft2)), np.abs(xfft2), color="blue")
    plt.plot(range(0, len(xfft3)), np.abs(xfft3), color="green")
    plt.title("cross-correlation output cross_correlate")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    xw_size = len(x) + len(y) - 1
    # The FFT is faster if the input signal is a power of 2.
    # fft_size = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    # fft_size =
    pad = len(y) - 1
    # pad = 0
    padded_x = (np.pad(x, (pad, pad), 'constant'))
    cc = ifft(fft(padded_x, fft_size) * np.conj(fft(y, fft_size)))
    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    cc = np.real(cc)
    # Calculate output spatial/time domain dimensions.
    out_size = len(x) + 2 * pad - len(y) + 1
    cc = cc[:out_size]
    import matplotlib.pyplot as plt
    plt.plot(range(0, len(cc)), cc)
    plt.title("cross-correlation output cross_correlate")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()
    return cc


def conv_forward_fft_1D_compress_perf(x, w, b, conv_param, index_back=0):
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
    :param energy_rate: how much energy should we preserve
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    # Grab conv parameters
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')
    if stride != 1:
        raise ValueError("convolution via fft can have stride only 1, but given: " + str(stride))
    N, C, W = x.shape
    F, C, WW = w.shape
    xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    half_fftsize = fftsize // 2
    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
    # Calculate output spatial/time domain dimensions.
    out_W = W + 2 * pad - WW + 1
    # Initialise the output.
    out = np.zeros([N, F, out_W])
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = fft(padded_x[nn, cc], fftsize)
                wfft = fft(w[ff, cc], fftsize)
                if index_back > 0:
                    index = half_fftsize - index_back
                    xfft = xfft[0:index]
                    wfft = wfft[0:index]
                outfft = xfft * np.conj(wfft)
                # we discarded half of the signal & probably close to zero coefficients so regain energy * 2
                outfft = np.real(ifft(np.pad(outfft, (0, fftsize - len(outfft)), 'constant'))) * 2
                if len(outfft) < out_W:
                    outfft = np.pad(outfft, (0, out_W - len(outfft)), 'constant')
                sum_out += outfft[:out_W]
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_fft_1D_compress_energy(x, w, b, conv_param, energy_rate=1.0):
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
    :param energy_rate: how much energy should we preserve
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    # Grab conv parameters
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')
    if stride != 1:
        raise ValueError("convolution via fft can have stride only 1, but given: " + str(stride))
    energy_rate_bound = (0.0, 1.0)
    if energy_rate < energy_rate_bound[0] or energy_rate > energy_rate_bound[1]:
        raise ValueError(
            "The energy rate should be between " + str(energy_rate_bound[0]) + " and " + str(energy_rate_bound[1]))
    N, C, W = x.shape
    F, C, WW = w.shape
    xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
    # Calculate output spatial/time domain dimensions.
    out_W = W + 2 * pad - WW + 1
    # Initialise the output.
    out = np.zeros([N, F, out_W])
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = fft(padded_x[nn, cc], fftsize)
                wfft = fft(w[ff, cc], fftsize)
                xfft = xfft[0:len(xfft) // 2]
                wfft = wfft[0: len(wfft) // 2]
                if energy_rate != 1.0:
                    squared_abs = np.abs(xfft) ** 2
                    full_energy = np.sum(squared_abs)
                    current_energy = 0
                    preserve_energy = full_energy * energy_rate
                    index = 0
                    while current_energy < preserve_energy and index < len(squared_abs) - 1:
                        current_energy += squared_abs[index]
                        index += 1
                    # print("index: ", index, " fft size: ", len(xfft))
                    if index == 0:
                        xfft = np.zeros(1)
                        wfft = np.zeros(1)
                    else:
                        xfft = xfft[0:index]
                        wfft = wfft[0:index]
                wfft = np.conj(wfft)
                outfft = xfft * wfft
                outfft = np.pad(outfft, (0, fftsize - len(outfft)), 'constant')
                outifft = ifft(outfft)
                out_real = np.real(outifft)
                if len(out_real) < out_W:
                    out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                # we discarded half of the signal & probably close to zero coefficients so regain energy * 2
                sum_out += out_real[:out_W] * 2
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_fft_1D_compress_optimized(x, w, b, conv_param, index_back=10):
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
    :return: a tuple of:
     - out: output data, of shape (N, W') where W' is given by:
     W' = 1 + (W + 2*pad - WW) / stride
     - cache: (x, w, b, conv_param)

     :see:  source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
     short: https://goo.gl/GwyhXz
    """
    # Grab conv parameters
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')
    if stride != 1:
        raise Exception("convolution via fft can have stride only 1, but given: " + str(stride))
    N, C, W = x.shape
    F, C, WW = w.shape
    xw_size = W + WW - 1
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int)
    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
    # Calculate output spatial/time domain dimensions.
    out_W = W + 2 * pad - WW + 1
    # Initialise the output.
    out = np.zeros([N, F, out_W])
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = fft(padded_x[nn, cc], fftsize)
                wfft = fft(w[ff, cc], fftsize)
                if index_back != None:
                    xfft = xfft[0:len(xfft) // 2]
                    xfft = xfft[0:-index_back]
                    wfft = wfft[0: len(wfft) // 2]
                    wfft = wfft[0:-index_back]
                wfft = np.conj(wfft)
                outfft = xfft * wfft
                outfft = np.pad(outfft, (0, fftsize - len(outfft)), 'constant')
                outifft = ifft(outfft)
                out_real = np.real(outifft)
                if len(out_real) < out_W:
                    out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                # we discarded half of the signal & probably close to zero coefficients so regain energy * 2
                sum_out += out_real[:out_W] * 2
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_fft_1D_compress(x, w, b, conv_param, index_back=10, fft_back=0):
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
    if stride != 1:
        raise ValueError("convolution via fft requires stride = 1, but given: ", stride)

    N, C, W = x.shape
    F, C, WW = w.shape

    xw_size = W + WW - 1
    print("xw_size: ", xw_size)
    # The FFT is faster if the input signal is a power of 2.
    fftsize = 2 ** np.ceil(np.log2(xw_size)).astype(int) - fft_back
    print("fftsize: ", fftsize)
    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))

    # Calculate output spatial/time domain dimensions.
    out_W = W + 2 * pad - WW + 1

    # Initialise the output.
    out = np.zeros([N, F, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = np.fft.fft(padded_x[nn, cc], fftsize)
                init_size = len(xfft)
                # print("xfft length: ", len(xfft))
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(xfft)), np.abs(xfft))
                # plt.title("cross-correlation output 1D fft cross-correlation compressed xfft 1")
                # plt.xlabel('time')
                # plt.ylabel('Amplitude')
                # plt.show()
                # print("first xfft: ", xfft)
                # xfft = xfft[:len(xfft) // 2]
                if index_back != None:
                    # index = len(xfft) // 2 - index_back
                    xfft = xfft[0:len(xfft) // 2]
                    xfft = np.concatenate((xfft[0:-index_back], np.zeros(index_back)))
                    # xfft = xfft[0: index + 1]
                    # squared_abs = np.abs(xfft) ** 2
                    # full_energy = np.sum(squared_abs)
                    # current_energy = 0.0
                    # preserve_energy = full_energy * preserve_energy_rate
                    # index = 0
                    # while current_energy < preserve_energy and index < len(squared_abs):
                    #     current_energy += squared_abs[index]
                    #     index += 1
                    # # print("index: ", index)
                    # xfft = xfft[:index]
                print("xfft: ", xfft)
                # xfft = xfft[:xfft.shape[0] // 2, :xfft.shape[1] // 2]
                print("length of xfft: ", len(xfft))
                filters = w[ff, cc]
                # print("filters: ", filters)
                # print("last shape of xfft: ", xfft.shape[-1])
                # The convolution theorem takes the duration of the response to be the same as the period of the data.
                filterfft = np.fft.fft(filters, fftsize)
                filterfft = np.conj(filterfft)
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(filterfft)), np.abs(filterfft))
                # plt.title("cross-correlation output 1D fft cross-correlation compressed filterfft1")
                # plt.xlabel('time')
                # plt.ylabel('Amplitude')
                # plt.show()
                if index_back != None:
                    index = len(filterfft) // 2 - index_back
                    filterfft = filterfft[0: len(filterfft) // 2]
                    filterfft = np.concatenate((filterfft[0:-index_back], np.zeros(index_back)))
                # filterfft = np.fft.fft(filters, xfft.shape[-1]*2)
                # filterfft = filterfft[:filterfft.shape[0] // 2, :filterfft.shape[1] // 2]
                # filterfft = filterfft[:filterfft.shape[-1] // 2]
                # print("filterfft: ", filterfft)
                # filterfft = np.conj(filterfft)
                # if index_back != None:
                #     xfft = np.concatenate((xfft, np.conj(np.flip(xfft[1:-1], axis=0))))
                #     filterfft = np.concatenate((filterfft, np.conj(np.flip(filterfft[1:-1], axis=0))))
                #     print("size of reconstructed xfft: ", len(xfft))
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(xfft)), np.abs(xfft))
                # plt.title("reconstructed xfft")
                # plt.xlabel('time')
                # plt.ylabel('Amplitude')
                # plt.show()
                #
                # import matplotlib.pyplot as plt
                # plt.plot(range(0, len(filterfft)), np.abs(filterfft))
                # plt.title("reconstructed filterfft")
                # plt.xlabel('time')
                # plt.ylabel('Amplitude')
                # plt.show()

                # xfft = xfft / norm(xfft)
                # filterfft = filterfft / norm(filterfft)

                outfft = xfft * filterfft
                # if index_back != 0:
                #     outfft = np.concatenate((outfft, np.conj(np.flip(outfft, axis=0))))
                # take the inverse of the output from the frequency domain and return the modules of the complex numbers
                # outfft = np.concatenate((outfft, np.zeros(init_size // 2)))

                # print("outfft size: ", outfft)
                outfft = np.pad(outfft, (0, init_size - len(outfft)), 'constant')
                outifft = np.fft.ifft(outfft)
                # out[nn, ff] += np.abs(np.fft.ifft2(xfft * filterfft, (out_H, out_W)))
                # outdouble = np.array(outifft, np.double)
                out_real = np.real(outifft)
                # out_real = np.abs(outifft)
                if len(out_real) < out_W:
                    out_real = np.pad(out_real, (0, out_W - len(out_real)), 'constant')
                # we cut off half of the signal and discarded some probably close to zero coefficients
                out_real = out_real[:out_W] * 2
                import matplotlib.pyplot as plt
                plt.plot(range(0, len(out_real)), out_real)
                plt.title("cross-correlation compressed")
                plt.xlabel('time')
                plt.ylabel('Amplitude')
                plt.show()

                sum_out += out_real[:out_W]

            # import matplotlib.pyplot as plt
            # plt.plot(range(0, len(sum_out)), sum_out)
            # plt.title("cross-correlation output 1D fft cross-correlation compressed")
            # plt.xlabel('time')
            # plt.ylabel('Amplitude')
            # plt.show()
            # crop the output to the expected shape
            # print("shape of expected resuls: ", out[nn, ff].shape)
            # print("shape of sum_out: ", sum_out.shape)
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_forward_fft_1D(x, w, b, conv_param, preserve_energy_rate=1.0):
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
    # print("fftsize my cross-correlation: ", fftsize)

    # Zero pad our tensor along the spatial dimensions.
    # Do not pad N (0,0) and C (0,0) dimensions, but only the 1D array - the W dimension (pad, pad).
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))

    # Calculate output spatial/time domain dimensions.
    out_W = W + 2 * pad - WW + 1

    # Initialise the output.
    out = np.zeros([N, F, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each time-series in the input batch.
        for ff in range(F):  # For each filter in w
            sum_out = np.zeros([out_W])
            for cc in range(C):
                xfft = np.fft.fft(padded_x[nn, cc], fftsize)
                # print("first xfft: ", xfft)
                # xfft = xfft[:len(xfft) // 2]
                if preserve_energy_rate < 1.0:
                    squared_abs = np.abs(xfft) ** 2
                    full_energy = np.sum(squared_abs)
                    current_energy = 0.0
                    preserve_energy = full_energy * preserve_energy_rate
                    index = 0
                    while current_energy < preserve_energy and index < len(squared_abs):
                        current_energy += squared_abs[index]
                        index += 1
                    # print("index: ", index)
                    xfft = xfft[:index]
                # print("xfft: ", xfft)
                # xfft = xfft[:xfft.shape[0] // 2, :xfft.shape[1] // 2]
                # print("xfft shape: ", xfft.shape)
                filters = w[ff, cc]
                # print("filters: ", filters)
                # print("last shape of xfft: ", xfft.shape[-1])
                # The convolution theorem takes the duration of the response to be the same as the period of the data.
                filterfft = np.fft.fft(filters, xfft.shape[-1])
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

            # import matplotlib.pyplot as plt
            # plt.plot(range(0, len(sum_out)), sum_out)
            # plt.title("cross-correlation output full 1D fft cross-correlation")
            # plt.xlabel('time')
            # plt.ylabel('Amplitude')
            # plt.show()
            # crop the output to the expected shape
            # print("shape of expected resuls: ", out[nn, ff].shape)
            # print("shape of sum_out: ", sum_out.shape)
            out[nn, ff] = sum_out + b[ff]

    cache = (x, w, b, conv_param)
    return out, cache