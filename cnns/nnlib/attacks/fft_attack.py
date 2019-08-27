from foolbox.attacks.base import Attack, call_decorator
import numpy as np
from cnns.nnlib.robustness.channels.channels_definition import fft_numpy
from cnns.nnlib.robustness.channels.channels_definition import fft_zero_values
from cnns.nnlib.robustness.channels.channels_definition import \
    fft_zero_low_magnitudes
from cnns.nnlib.robustness.channels.channels_definition import \
    replace_frequencies_numpy
import torch
from cnns.nnlib.pytorch_layers.pytorch_utils import get_xfft_hw
from cnns.nnlib.pytorch_layers.pytorch_utils import get_ifft_hw
from cnns.nnlib.pytorch_layers.pytorch_utils import get_max_min_complex
from cnns.nnlib.pytorch_layers.pytorch_utils import get_sorted_spectrum_indices
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

nprng = np.random.RandomState()
nprng.seed(37)


def pytorch_net(net):
    """
    :param net: a Pytorch model
    :return: Pytorch network model whose predictions are returned as numpy array
    """

    def net_wrapper(input):
        predictions = net(input)
        return predictions.detach().cpu().numpy()

    return net_wrapper


def bisearch_to_decrease_rate(input, label, func, net, low=0, high=100,
                              resolution=1.0):
    last_adv_image = None
    last_compression_rate = None

    while low <= high:
        mid = (high + low) / 2
        adv_image = func(input, mid)
        predictions = net(adv_image)
        predicted_class_id = np.argmax(predictions)

        if predicted_class_id != label:
            last_adv_image = adv_image
            last_compression_rate = mid
            # binary search
            high = mid - resolution
        else:
            low = mid + resolution
    return last_adv_image, last_compression_rate


def bisearch_to_increase_rate(input, label, func, net, low=0, high=100.0,
                              resolution=1.0):
    last_compression_rate = None
    last_adv_image = None

    while low <= high:
        mid = (high + low) / 2
        adv_image = func(input, mid)
        predictions = net(adv_image)
        predicted_class_id = np.argmax(predictions)

        if predicted_class_id != label:
            last_adv_image = adv_image
            last_compression_rate = mid
            # binary search
            low = mid + resolution
        else:
            high = mid - resolution
    return last_adv_image, last_compression_rate


class FFTHighFrequencyAttack(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, compress_rate=50):
        return fft_numpy(numpy_array=input_or_adv, compress_rate=compress_rate)


class FFTHighFrequencyAttackAdversary(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, net=None):
        adv_image, _ = bisearch_to_decrease_rate(input=input_or_adv,
                                                 label=label, net=net,
                                                 func=fft_numpy)
        return adv_image


class FFTLimitFrequencyAttack(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, net=None,
                 compress_rate=50, compress_resolution=1.0):
        """
        Binary search for the highest inverse_compress_rate so that we can
        recover as many high frequency coefficient as possible.

        :param input_or_adv: the adversarial image
        :param label: the correct label
        :param unpack: not used
        :param net: the ml model
        :param compress_rate: how much to compress
        :return: an adversarial image
        """

        def func(image, rate):
            return fft_numpy(numpy_array=image,
                             compress_rate=compress_rate,
                             inverse_compress_rate=rate)

        adv_image, _ = bisearch_to_increase_rate(input_or_adv,
                                                 label=label,
                                                 func=func,
                                                 net=net,
                                                 high=compress_rate)
        return adv_image


class FFTLimitFrequencyAttackAdversary(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, net=None,
                 compress_resolution=1.0):
        """
        Binary search for the highest inverse_compress_rate so that we can
        recover as many high frequency coefficient as possible.

        :param input_or_adv: the adversarial image
        :param label: the correct label
        :param unpack: not used
        :param net: the ml model
        :param compress_rate: how much to compress
        :return: an adversarial image
        """
        adv_image, compress_rate = bisearch_to_decrease_rate(input=input_or_adv,
                                                             label=label,
                                                             net=net,
                                                             func=fft_numpy)

        if adv_image is None:
            return None

        def func(image, rate):
            return fft_numpy(numpy_array=image,
                             compress_rate=compress_rate,
                             inverse_compress_rate=rate)

        adv_image2, _ = bisearch_to_increase_rate(input_or_adv,
                                                  label=label,
                                                  func=func,
                                                  net=net,
                                                  high=compress_rate)
        if adv_image2 is not None:
            adv_image = adv_image2

        return adv_image


class FFTReplaceFrequencyAttack(Attack):

    def is_adv(self, label, predicated_label, target_label):
        # For totally un-targeted attack we need the predicated label to be
        # differfent from the original label.
        is_adv_ = (predicated_label != label)
        # The predicated label has to be the same as the target_label:
        if target_label is not None and predicated_label != target_label:
            is_adv_ = False
        return is_adv_

    def __call__(self, input_or_adv, label=None, unpack=True, net=None,
                 compress_resolution=1.0, input2=None, target_label=None,
                 is_next_power2=False, high=True):
        """
        Replace some frequencies in input_or_adv with some frequencies from
        input2.

        :param input_or_adv: the original image
        :param label: the ground truth label of the original image
        :param unpack: maintained to adhere to the Attack interface, not used
        :param net: the neural network, we need only the output of classification
        :param compress_resolution: for FFT, granularity of compression
        :param input2: the other image from which we take some frequencies
        :param target_label: the label id of the input2 image
        :param is_next_power2: should we add padding to the nearest power of 2
        :param replace high or low frequencies
        :return: the adv_image found
        """
        adv_image = None
        last_adv_mid = None
        low = 0
        high = 100

        # Find to which of the lowest frequencies we have to replace in the
        # original image with the frequencies from the other image to get an
        # adversarial example.
        # We aim at the label which is assigned to input2. This is a type of
        # targeted attack.
        while low <= high:
            mid = (high + low) / 2
            image = replace_frequencies_numpy(
                input_to=input_or_adv, input_from=input2, compress_rate=mid,
                is_next_power2=is_next_power2, high=high)
            predictions = net(image)
            predicated_label = np.argmax(predictions)

            is_adv = self.is_adv(label=label, predicated_label=predicated_label,
                                 target_label=target_label)

            if is_adv:
                adv_image = image
                last_adv_mid = mid
                # binary search: minimize the max compression (replacement)
                high = mid - compress_resolution
            else:
                low = mid + compress_resolution

        # The default class for no values can be the ground-truth class for this
        # image.
        if adv_image is None:
            return None

        # Restore some high frequencies from the original image.
        # Optimize the low point - for which of the higher frequencies (going
        # down to lower frequencies) we can descend and recover the original
        # frequencies.
        low = 0
        high = last_adv_mid
        while low <= high:
            mid = (high + low) / 2
            image = replace_frequencies_numpy(
                input_to=adv_image, input_from=input_or_adv,
                compress_rate=mid, high=high)
            predictions = net(image)
            predicated_label = np.argmax(predictions)

            is_adv = self.is_adv(label=label, predicated_label=predicated_label,
                                 target_label=target_label)

            if is_adv:
                adv_image = image
                # binary search: maximize the mid compression
                low = mid + compress_resolution
            else:
                high = mid - compress_resolution

        return adv_image


class FFTSingleFrequencyAttack(Attack):
    """Perturbs just a single frequency and sets it to the min or max."""

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 max_frequencies=1000):

        """Perturbs just a single frequency and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        # Give the axis for the color channel.
        channel_axis = a.channel_axis(batch=False)
        assert channel_axis == 0
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        H = image.shape[axes[0]]
        W = image.shape[axes[1]]

        image_torch = torch.from_numpy(image).unsqueeze(
            0)  # we need the batch dim
        is_next_power2 = False
        onesided = True
        xfft, H_fft, W_fft = get_xfft_hw(
            input=image_torch, is_next_power2=is_next_power2, onesided=onesided)
        maxf, minf = get_max_min_complex(xfft=xfft)
        W_xfft = xfft.shape[-2]
        total_freqs = H_fft * W_xfft
        freqs = nprng.permutation(total_freqs)
        # freqs = freqs[:max_frequencies]
        for i, freq in enumerate(freqs):
            w = freq % W_xfft
            h = freq // W_xfft

            location = [h, w]
            # Add the channel dimension.
            location.insert(channel_axis, slice(None))
            # Add the batch dim.
            location.insert(0, slice(None))
            location = tuple(location)

            for value in [minf, maxf]:
                perturbed_xfft = xfft.clone()
                value = check_real_vals(
                    H_fft=H_fft, W_fft=W_fft, h=h, w=w, value=value)
                perturbed_xfft[location] = value
                perturbed = get_ifft_hw(
                    xfft=perturbed_xfft, H_fft=H_fft, W_fft=W_fft, H=H, W=W)
                perturbed = perturbed.detach().cpu().numpy().squeeze()
                _, is_adv = a.predictions(perturbed)
                if is_adv:
                    return


class FFTMultipleFrequencyAttack(Attack):
    """Perturbs multiple frequency coefficients and sets them to the min or
    max frequency coefficient."""

    def __init__(self, args, model=None, criterion=Misclassification(),
                 distance=MSE, threshold=None, max_frequencies_percent=30,
                 iterations=100, is_strict=True, is_debug=True, is_fast=False):
        super(FFTMultipleFrequencyAttack, self).__init__(
            model=model, criterion=criterion, distance=distance,
            threshold=threshold)
        self.args = args
        self.max_frequencies_percent = max_frequencies_percent
        self.iterations = iterations
        self.is_strict = is_strict
        self.is_debug = is_debug
        self.is_fast = is_fast

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Perturbs just a single frequency and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        # Give the axis for the color channel.
        channel_axis = a.channel_axis(batch=False)
        assert channel_axis == 0
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        H = image.shape[axes[0]]
        W = image.shape[axes[1]]

        image_torch = torch.from_numpy(image).unsqueeze(
            0)  # we need the batch dim
        is_next_power2 = False
        onesided = True
        xfft, H_fft, W_fft = get_xfft_hw(
            input=image_torch, is_next_power2=is_next_power2, onesided=onesided)
        # maxf, minf = get_max_min_complex(xfft=xfft)
        value = torch.tensor([0.0, 0.0])
        W_xfft = xfft.shape[-2]
        total_freqs = H_fft * W_xfft
        max_frequencies = int(total_freqs * self.max_frequencies_percent / 100)
        for iter in range(self.iterations):
            freqs = nprng.permutation(total_freqs)
            freqs = freqs[:max_frequencies]
            perturbed_xfft = xfft.clone()
            for num_freqs, freq in enumerate(freqs):
                w = freq % W_xfft
                h = freq // W_xfft

                location = [h, w]
                # Add the channel dimension.
                location.insert(channel_axis, slice(None))
                # Add the batch dim.
                location.insert(0, slice(None))
                location = tuple(location)

                # if np.random.randint(0, 2) == 1:
                #     value = minf
                # else:
                #     value = maxf
                # value = check_real_vals(
                #     H_fft=H_fft, W_fft=W_fft, h=h, w=w, value=value)
                perturbed_xfft[location] = value
                perturbed = get_ifft_hw(
                    xfft=perturbed_xfft, H_fft=H_fft, W_fft=W_fft, H=H, W=W)
                perturbed = perturbed.detach().cpu().numpy().squeeze()
                if self.is_strict:
                    perturbed = np.clip(perturbed, a_min=self.args.min,
                                        a_max=self.args.max)
                _, is_adv, _, dist = a.predictions(
                    perturbed, return_details=True)
                if is_adv:
                    if self.is_debug:
                        dist = np.sqrt(dist.value)
                        print(f'iterations: {iter}, '
                              f'number of modified frequencies: {num_freqs}, '
                              f'dist: {dist}')
                    if self.is_fast:
                        return
                    break


class FFTMultipleFrequencyBinarySearchAttack(Attack):
    """Perturbs multiple frequency coefficients and sets them to the min or
    max frequency coefficient. In each iteration of the algorithm, we binary
    search what is the minimum number of coefficients to be changed."""

    def __init__(self, args, model=None, criterion=Misclassification(),
                 distance=MSE, threshold=None, iterations=100, is_strict=True,
                 is_debug=True, is_fast=False, resolution=1):
        super(FFTMultipleFrequencyBinarySearchAttack, self).__init__(
            model=model, criterion=criterion, distance=distance,
            threshold=threshold)
        self.args = args
        self.iterations = iterations
        self.is_strict = is_strict
        self.is_debug = is_debug
        self.is_fast = is_fast
        self.resolution = resolution

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Perturbs just a single frequency and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            to preserve the inheritance requirement.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        # Give the axis for the color channel.
        channel_axis = a.channel_axis(batch=False)
        assert channel_axis == 0
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        H = image.shape[axes[0]]
        W = image.shape[axes[1]]
        # We need the batch dim.
        image_torch = torch.from_numpy(image).unsqueeze(0)
        is_next_power2 = False
        onesided = True
        xfft, H_fft, W_fft = get_xfft_hw(
            input=image_torch, is_next_power2=is_next_power2, onesided=onesided)
        # maxf, minf = get_max_min_complex(xfft=xfft)
        value = torch.tensor([0.0, 0.0])
        W_xfft = xfft.shape[-2]
        total_freqs = H_fft * W_xfft
        low = 0
        high = total_freqs
        for iter in range(self.iterations):
            freqs = nprng.permutation(total_freqs)
            while low <= high:
                # What is the percentage of modified frequencies?
                mid = (low + high) // 2
                freqs = freqs[:mid]
                perturbed_xfft = xfft.clone()
                for num_freqs, freq in enumerate(freqs):
                    w = freq % W_xfft
                    h = freq // W_xfft
                    perturbed_xfft[:, :, h, w] = value
                perturbed = get_ifft_hw(
                    xfft=perturbed_xfft, H_fft=H_fft, W_fft=W_fft, H=H, W=W)
                perturbed = perturbed.detach().cpu().numpy().squeeze()
                if self.is_strict:
                    perturbed = np.clip(perturbed, a_min=self.args.min,
                                        a_max=self.args.max)
                _, is_adv, _, dist = a.predictions(
                    perturbed, return_details=True)
                if is_adv:
                    high = mid - self.resolution
                    if self.is_debug:
                        dist = np.sqrt(dist.value)
                        print(f'iterations: {iter}, '
                              f'number of modified frequencies: {num_freqs}, '
                              f'dist: {dist}')
                else:
                    low = mid + self.resolution


def check_real_vals(H_fft, W_fft, h, w, value):
    """
    Check if x,y coordinates are subject to the real value constraint. If so,
    add the imaginary part to the real one, and set the imaginary part to 0.

    :param xfft: the input fft map
    :param H_fft: the height of the fft map
    :param W_fft: the width of the fft map
    :param x: the x coordinate to be modified
    :param y: the y coordinate to be modified
    :param value:
    :return: the value modified if x,y are subject to the real constraint
    """
    # Maintain real values.
    is_real = False  # Is the location for the real valued constraint?
    # Top-left corner.
    if h == 0 and w == 0:
        is_real = True
    # Even sizes.
    if H_fft % 2 == 0:
        assert W_fft % 2 == 0
        if h == H_fft // 2 + 1:
            # Middle-left element.
            if w == 0:
                is_real = True
            # Middle-right element.
            elif w == W_fft // 2 + 1:
                is_real = True
        # Top-right element.
        elif h == 0 and w == W_fft // 2 + 1:
            is_real = True
    if is_real:
        value[0] += value[1]
        value[1] = 0.0  # set the imaginary part to 0

    return value


class FFTSmallestFrequencyAttack(Attack):
    """Starts from the smallest frequency magnitudes and set the frequency
    coefficients to 0.
    This is a slower version of FFTLimitMagnitudesAttack.
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 max_frequencies=100000, debug=True):

        """Sets the smallest frequency coefficients in their magnitudes to 0.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        # Give the axis for the color channel.
        channel_axis = a.channel_axis(batch=False)
        assert channel_axis == 0
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        H = image.shape[axes[0]]
        W = image.shape[axes[1]]

        image_torch = torch.from_numpy(image).unsqueeze(
            0)  # we need the batch dim
        is_next_power2 = False
        onesided = True
        xfft, H_fft, W_fft = get_xfft_hw(
            input=image_torch, is_next_power2=is_next_power2, onesided=onesided)
        freqs = get_sorted_spectrum_indices(xfft=xfft)
        W_xfft = xfft.shape[-2]
        perturbed_xfft = xfft.clone()
        channel_size = H_fft * W_xfft
        zero_value = torch.tensor([0.0, 0.0])
        last_magnitude = 0.0
        # freqs = freqs[:max_frequencies]
        for i, freq in enumerate(freqs):
            c = freq // channel_size
            w = freq % W_xfft
            h = (freq - c * channel_size) // W_xfft

            location = [c, h, w]
            # Add the batch dim.
            location.insert(0, 0)
            location = tuple(location)
            if debug:
                elem = perturbed_xfft[location]
                re = elem[0]
                im = elem[1]
                magnitude = np.sqrt(re ** 2 + im ** 2)
                assert magnitude >= last_magnitude
                last_magnitude = magnitude
            perturbed_xfft[location] = zero_value
            perturbed = get_ifft_hw(
                xfft=perturbed_xfft, H_fft=H_fft, W_fft=W_fft, H=H, W=W)
            perturbed = perturbed.detach().cpu().numpy().squeeze()
            _, is_adv = a.predictions(perturbed)
            if is_adv:
                if debug:
                    print('# of frequencies zeroed out: ', i + 1)
                return


class FFTLimitValuesAttack(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, net=None):
        """
        Binary search for magnitudes to zero out.

        :param input_or_adv: the adversarial image
        :param label: the correct label
        :param unpack: not used
        :param net: the ml model
        :return: an adversarial image
        """
        onesided = True
        is_next_power2 = False
        net = self._default_model._model  # we operate directly in Pytorch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        net.to(device)
        net = pytorch_net(net)
        input = torch.tensor(input_or_adv).unsqueeze(dim=0).to(device)
        xfft, _, _ = get_xfft_hw(input=input, onesided=onesided,
                                 is_next_power2=is_next_power2)
        spectrum = get_spectrum(xfft, squeeze=False)
        min = spectrum.min()
        max = spectrum.max()

        def decrease_func(image, high):
            return fft_zero_values(
                input=image, high=high, low=min, is_next_power2=is_next_power2,
                onesided=onesided)

        adv_image, high = bisearch_to_decrease_rate(input=input,
                                                    label=label,
                                                    net=net,
                                                    low=min,
                                                    high=max,
                                                    func=decrease_func)

        if adv_image is None:
            return None

        def increase_func(image, low):
            return fft_zero_values(
                input=image, high=high, low=low, is_next_power2=is_next_power2,
                onesided=onesided)

        adv_image2, _ = bisearch_to_increase_rate(input=input,
                                                  label=label,
                                                  net=net,
                                                  low=min,
                                                  high=high,
                                                  func=increase_func)
        if adv_image2 is not None:
            adv_image = adv_image2

        return adv_image.detach().squeeze().cpu().numpy()


class FFTLimitMagnitudesAttack(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, net=None):
        """
        Binary search for magnitudes to zero out.

        :param input_or_adv: the adversarial image
        :param label: the correct label
        :param unpack: not used
        :param net: the ml model
        :return: an adversarial image
        """
        onesided = True
        is_next_power2 = False
        net = self._default_model._model  # we operate directly in Pytorch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        net.to(device)
        net = pytorch_net(net)
        input = torch.tensor(input_or_adv).unsqueeze(dim=0).to(device)
        xfft, _, _ = get_xfft_hw(input=input, onesided=onesided,
                                 is_next_power2=is_next_power2)
        spectrum = get_spectrum(xfft, squeeze=False)
        min = spectrum.min()
        max = spectrum.max()

        def decrease_func(image, high):
            return fft_zero_low_magnitudes(
                input=image, high=high, low=min, is_next_power2=is_next_power2,
                onesided=onesided)

        _, high = bisearch_to_decrease_rate(input=input,
                                            label=label,
                                            net=net,
                                            low=min,
                                            high=max,
                                            func=decrease_func)

        if high is None:
            return None

        def increase_func(image, low):
            return fft_zero_low_magnitudes(
                input=image, high=high, low=low, is_next_power2=is_next_power2,
                onesided=onesided)

        adv_image, _ = bisearch_to_increase_rate(input=input,
                                                 label=label,
                                                 net=net,
                                                 low=min,
                                                 high=high,
                                                 func=increase_func)
        if adv_image is None:
            return None
        else:
            return adv_image.detach().squeeze().cpu().numpy()
