# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import logging
import functools
import torch
from foolbox.attacks.base import call_decorator
from foolbox.attacks.carlini_wagner import CarliniWagnerL2Attack
from foolbox.attacks.carlini_wagner import AdamOptimizer
from foolbox.adversarial import Adversarial
from cnns.nnlib.pytorch_layers.fft_band_2D_complex_mask import \
    FFTBandFunctionComplexMask2D
from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBandFunction2D
from foolbox.criteria import Misclassification
from foolbox.distances import MSE
from cnns.nnlib.utils.complex_mask import get_disk_mask
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack
from cnns.nnlib.robustness.randomized_defense import defend
from foolbox.attacks.additive_noise import AdditiveGaussianNoiseAttack


class CarliniWagnerL2AttackRoundFFT(CarliniWagnerL2Attack):
    """The L2 version of the Carlini & Wagner attack.

    This attack is described in [1]_. This implementation
    is based on the reference implementation by Carlini [2]_.
    For bounds ≠ (0, 1), it differs from [2]_ because we
    normalize the squared L2 loss with the bounds.

    References
    ----------
    .. [1] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644
    .. [2] https://github.com/carlini/nn_robust_attacks

    """

    def __init__(self, args, model=None, criterion=Misclassification(),
                 distance=MSE, threshold=None, get_mask=get_hyper_mask):
        super(CarliniWagnerL2AttackRoundFFT, self).__init__(
            model=model, criterion=criterion, distance=distance,
            threshold=threshold)
        if args is None:
            raise Exception("args have to be provided!")
            # from cnns.nnlib.datasets.cifar import cifar_mean_array
            # from cnns.nnlib.datasets.cifar import cifar_std_array
            # self.std_array = cifar_std_array
            # self.mean_array = cifar_mean_array
        else:
            self.args = args
            self.std_array = args.std_array
            self.mean_array = args.mean_array
            self.rounder = DenormRoundNorm(
                mean_array=args.mean_array, std_array=args.std_array,
                values_per_channel=args.values_per_channel)
            self.get_mask = get_mask
            self.noise = AdditiveUniformNoiseAttack()
            self.gauss = AdditiveGaussianNoiseAttack()

    def fft_complex_compression(self, image, is_clip=False, ctx=None,
                                onesided=True):
        fft_image = FFTBandFunctionComplexMask2D.forward(
            ctx=ctx,
            input=torch.from_numpy(image).unsqueeze(0),
            compress_rate=self.args.compress_fft_layer, val=0,
            interpolate=self.args.interpolate,
            get_mask=self.get_mask,
            onesided=onesided).numpy().squeeze(0)
        if is_clip:
            return np.clip(fft_image, a_min=self.args.min, a_max=self.args.max)
        else:
            return fft_image

    def fft_lshape_compression(self, image, is_clip=False, ctx=None,
                               onesided=True):
        fft_image = FFTBandFunction2D.forward(
            ctx=ctx,
            input=torch.from_numpy(image).unsqueeze(0),
            compress_rate=self.args.compress_fft_layer,
            onesided=onesided).numpy().squeeze()
        if is_clip:
            return np.clip(fft_image, a_min=self.args.min, a_max=self.args.max)
        else:
            return fft_image

    def init_roundfft_adversarial(self, original_image, original_class):
        """
        Initialize the state of the adversarial object to save the adversarial
        examples that break the rounding defense.

        :param rounded_image: current image to be adversarial against the
        rounding defense
        :param original_class: the class (label) for the original image
        """
        model = self._default_model
        criterion = self._default_criterion
        distance = self._default_distance
        threshold = self._default_threshold
        if model is None or criterion is None:
            raise ValueError('The attack needs to be initialized'
                             ' with a model and a criterion or it'
                             ' needs to be called with an Adversarial'
                             ' instance.')
        image = original_image
        image = self.add_distortion(image)

        # TODO: should we check if the transformed image remains correctly
        #  classified?

        self.roundfft_adversarial = Adversarial(
            model=model, criterion=criterion, original_image=image,
            original_class=original_class, distance=distance,
            threshold=threshold)

    def add_distortion(self, image, is_clip=False):
        """
        Adds rounding, fft, uniform and gaussian noise distortions.

        :param image: the input image
        :return: the distorted image
        """
        if self.args.values_per_channel > 0:
            image = self.rounder.round(image)
            if is_clip:
                image = np.clip(image, a_min=self.args.min, a_max=self.args.max)
        if self.args.compress_fft_layer > 0:
            # clip is done inside fft compression.
            image = self.fft_complex_compression(image, is_clip=is_clip)
        if self.args.noise_epsilon > 0:
            noise = self.noise._sample_noise(
                epsilon=self.args.noise_epsilon, image=image,
                bounds=(self.args.min, self.args.max))
            image += noise
            if is_clip:
                image = np.clip(image, a_min=self.args.min, a_max=self.args.max)
        if self.args.noise_sigma > 0:
            noise = self.gauss._sample_noise(
                epsilon=self.args.noise_sigma, image=image,
                bounds=(self.args.min, self.args.max))
            image += noise
            if is_clip:
                image = np.clip(image, a_min=self.args.min, a_max=self.args.max)

        return image

    def get_roundfft_adversarial(self):
        """

        :return: the current best adversarial against the rounding defnese or
        None if no such adversarial was found.
        """
        return self.roundfft_adversarial.image

    def rounded_fft_predictions(self, adversarial, image_attack):
        """
        Check if the perturbed image is still adversarial after rounding and the
        fft prediction.

        :param image_attack: the perturbed image
        :return: predictions, is_adv
        """
        if self.args.values_per_channel > 0:
            image_attack = self.rounder.round(image_attack)
        if self.args.compress_fft_layer > 0:
            image_attack = self.fft_complex_compression(image_attack)
        return adversarial.predictions(image_attack)

    def rounded_predictions(self, adversarial, image_attack):
        """
        Check if the perturbed image is still adversarial after rounding.

        :param image_attack: the perturbed image
        :return: predictions, is_adv
        """
        if self.args.values_per_channel > 0:
            image_attack = self.rounder.round(image_attack)
        return adversarial.predictions(image_attack)

    def attack(call_fn):
        @functools.wraps(call_fn)
        def wrapper(self, input_or_adv, label, unpack=True, **kwargs):
            """
            Attack the model starting from the original_image by perturbing it.
            The same params as in the __call__ method.
            :return: the adversarial image that fools the model that was created
            from the CarliniWagnerL2 attack and then rounded. The final rounded
            image is adversarial.
            """
            self.init_roundfft_adversarial(original_image=input_or_adv,
                                           original_class=label)
            original_adversarial = call_fn(
                self, input_or_adv, label=label, **kwargs)

            if unpack:
                return self.get_roundfft_adversarial()
            else:
                return original_adversarial, self.roundfft_adversarial

        return wrapper

    @attack
    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search_steps=5, max_iterations=1000, confidence=0,
                 learning_rate=5e-3, initial_const=1e-2, abort_early=True):

        """The L2 version of the Carlini & Wagner attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search_steps : int
            The number of steps for the binary search used to
            find the optimal tradeoff-constant between distance and confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more
            accurate; setting it too small will require a large learning rate
            and will produce poor results.
        confidence : int or float
            Confidence of adversarial examples: a higher value produces
            adversarials that are further away, but more strongly classified
            as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If `binary_search_steps`
            is large, the initial constant is not important.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased
            for some time (a tenth of max_iterations).
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            logging.fatal('Applied gradient-based attack to model that '
                          'does not provide gradients.')
            return

        min_, max_ = a.bounds()

        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(x):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            x = np.tanh(x)

            grad = 1 - np.square(x)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = x * b + a

            grad = grad * b
            return x, grad

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_original = to_attack_space(a.original_image)

        # will be close but not identical to a.original_image
        reconstructed_original, _ = to_model_space(att_original)

        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_const
        lower_bound = 0
        upper_bound = np.inf

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                # TODO: find out why... it's not obvious why this is useful
                const = upper_bound

            logging.info('starting optimization with const = {}'.format(const))

            att_perturbation = np.zeros_like(att_original)

            # create a new optimizer to minimize the perturbation
            optimizer = AdamOptimizer(att_perturbation.shape)

            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf

            for iteration in range(max_iterations):
                x, dxdp = to_model_space(att_original + att_perturbation)

                # We try to find the rounded adversarial in parallel to finding
                # the normal adversarial for this attack.
                # _, is_round_adv = self.rounded_predictions(
                #     image_attack=x, values_per_channel=values_per_channel)
                # logits, is_round_adv = self.rounded_predictions(
                #     adversarial=a, image_attack=x,
                #     values_per_channel=values_per_channel)
                # logits, is_adv = a.predictions(x)

                # x = np.clip(x, a_min=self.args.min, a_max=self.args.max)
                x_prime = x
                x_prime = self.add_distortion(x_prime)

                if self.args.noise_iterations > 0:
                    # This is the randomized defense.
                    result_noise, predictions = defend(
                        image=x_prime,
                        fmodel=self._default_model,
                        args=self.args)

                    in_bounds = self.roundfft_adversarial.in_bounds(x_prime)
                    if in_bounds is False:
                        raise Exception("Not in bounds")
                    # print("dir self.roundfft_adversarial: ", dir(self.roundfft_adversarial))
                    if self.args.use_logits_random_defense:
                        is_adv, _, _ = self.roundfft_adversarial._Adversarial__is_adversarial(
                            x_prime, predictions, in_bounds)
                    else:
                        is_adv = result_noise.class_id != a.original_class
                        # Update the adversarial for the randomized version of the image.
                        self.roundfft_adversarial.predictions(x_prime)
                else:
                    # x_prime = np.clip(x_prime, self.args.min, self.args.max)
                    # update the adversarial for the rounded version of the image
                    _, is_adv = self.roundfft_adversarial.predictions(x_prime)

                # print("diff between input image and rounded: ",
                #       np.sum(np.abs(x_rounded - x)))

                # the perturbations are with respect to the original image
                # logits, is_adv = a.predictions(x_rounded)
                # logits, is_adv = a.predictions(x)
                strict = True
                logits, _ = a.predictions(x_prime, strict=strict)

                # dldx - gradient of the loss with respect to input x
                loss, dldx = self.loss_function(
                    const, a, x, logits, reconstructed_original,
                    confidence, min_, max_, strict=strict)

                logging.info('loss: {}; best overall distance: {}'.format(
                    loss, a.distance))

                # backprop the gradient of the loss w.r.t. x further
                # to get the gradient of the loss w.r.t. att_perturbation
                assert dldx.shape == x.shape
                assert dxdp.shape == x.shape
                # we can do a simple elementwise multiplication, because
                # grad_x_wrt_p is a matrix of elementwise derivatives
                # (i.e. each x[i] w.r.t. p[i] only, for all i) and
                # grad_loss_wrt_x is a real gradient reshaped as a matrix
                gradient = dldx * dxdp

                att_perturbation += optimizer(gradient, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if abort_early and \
                        iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            if found_adv:
                logging.info('found adversarial with const = {}'.format(const))
                upper_bound = const
            else:
                logging.info('failed to find adversarial '
                             'with const = {}'.format(const))
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2
