from foolbox.attacks.base import Attack
import numpy as np
from cnns.nnlib.robustness.channels.channels_definition import fft_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    replace_high_frequencies_numpy

nprng = np.random.RandomState()
nprng.seed(31)


def bisearch_to_decrease_rate(input, label, func, net, low=0, high=100,
                              compress_resolution=1.0):
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
            high = mid - compress_resolution
        else:
            low = mid + compress_resolution
    return last_adv_image, last_compression_rate


def bisearch_to_increase_rate(input, label, func, net, low=0, high=100,
                              compress_resolution=1.0):
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
            low = mid + compress_resolution
        else:
            high = mid - compress_resolution
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
        _, compress_rate = bisearch_to_decrease_rate(input=input_or_adv,
                                                     label=label,
                                                     net=net,
                                                     func=fft_numpy)

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


class FFTReplaceFrequencyAttack(Attack):

    def is_adv(self, label, predicated_label, target_label):
        # For totally un-targeted attack we need the predicated label to be
        # differfent from the original label.
        is_adv_ = (predicated_label != label)
        # The predicated label has to be the same as the target_label:
        if target_label is not None and predicated_label != target_label:
            is_adv = False
        return is_adv_

    def __call__(self, input_or_adv, label=None, unpack=True, net=None,
                 compress_resolution=1.0, input2=None, target_label=None,
                 is_next_power2=False):
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
            image = replace_high_frequencies_numpy(
                input_to=input_or_adv, input_from=input2, compress_rate=mid,
                is_next_power2=is_next_power2)
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
            image = replace_high_frequencies_numpy(
                input_to=adv_image, input_from=input_or_adv,
                compress_rate=mid)
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
