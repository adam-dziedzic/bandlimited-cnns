from __future__ import division
import numpy as np

from foolbox.attacks.base import Attack
from foolbox.attacks.base import call_decorator
from foolbox.rngs import nprng


class MultiplePixelsAttack(Attack):
    """Perturbs multiple pixels and sets them to the min or max.

    Proposed by Adam Dziedzic (adam.cajf@gmail.com).

    """

    def perturb_pixel(self, perturbed, pixel, min_, max_, w, channel_axis = 0):
        """
        Perturb a
        :param perturbed: the image to be perturbed
        :param pixel: the pixel number (if the image was flattened)
        :param min_: the min value to be set
        :param max_: the max value to be set
        :param w: the width of the image
        :param channel_axis:
        :return:
        """
        x = pixel % w
        y = pixel // w

        location = [x, y]
        # set the same value in each channel
        location.insert(channel_axis, slice(None))
        location = tuple(location)

        if np.random.randint(0, 2) == 1:
            value = min_
        else:
            value = max_
        perturbed[location] = value

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 num_pixels=1000, iterations=1):

        """Perturbs multiple pixels and sets them to the min or max.

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
        num_pixels : int
            Number of pixels that are perturbed in a single trial.
        iterations : int
            Number of times to try different set of num_pixels until an
            adversarial example is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        channel_axis = a.channel_axis(batch=False)
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = image.shape[axes[0]]
        w = image.shape[axes[1]]

        min_, max_ = a.bounds()

        for _ in range(iterations):
            pixels = nprng.permutation(h * w)
            pixels = pixels[:num_pixels]

            perturbed = image.copy()
            for pixel in range(pixels):
                self.perturb_pixel(perturbed=perturbed,
                                   pixel=pixel,
                                   min_=min_,
                                   max_=max_,
                                   w = w,
                                   channel_axis=channel_axis)
                _, is_adv = a.predictions(perturbed)
                if is_adv:
                    return
