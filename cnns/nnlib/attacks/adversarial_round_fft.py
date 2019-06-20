from foolbox.adversarial import Adversarial
from foolbox.distances import MSE


class AdversarialRoundFFT(Adversarial):

    def __init__(
            self,
            model,
            criterion,
            original_image,
            original_class,
            distance=MSE,
            threshold=None,
            verbose=False,
            strict=False):
        super(AdversarialRoundFFT, self).__init__(
            model=model, criterion=criterion, original_image=original_image,
            original_class=original_class, distance=distance,
            threshold=threshold, verbose=verbose, strict=strict)

    def predictionsRoundFFT(self, adv_transformed, adv, strict=False,
                            return_details=False):
        """Interface to model.predictions for attacks.

        If the adv_transformed (the image after FFT, rounding or noisy channel
        transformation) is misclassified, then the adversarial image is the
        input to the transformation. This pre-transformed image is kept as adv.
        The general pipeline is that we receive the standard non-adversarial
        image x. Then we modify image x via our attack to get an adversarial
        image (adv). Then, the adv is not immediately given into the classifier
        but transformed and then given into the classifier (we denote this image
        as adv_transformed). Thus, the attack has to return the adv image if the
        adversarial is found.

        Parameters
        ----------
        adv_transformed : `numpy.ndarray`
            Image with shape (height, width, channels) after rounding, FFT or
            noisy channel transformation.
        adv: `numpy.ndarray`
            The image before transformation. This is the image that after
            rounding, FFT or noisy channel becomes adversarial.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(adv)
        # assert not strict or in_bounds
        if strict and not in_bounds:
            min_, max_ = self.bounds()
            raise Exception("image not in bounds: ", " min: ",
                            adv_transformed.min(),
                            " max: ", adv_transformed.max(),
                            " min_bound: ", min_,
                            " max_bound: ", max_)

        self._total_prediction_calls += 1
        # The predictions are computed for the adversarial example after the
        # transformation (for example, via the randomized/noisy channel or after
        # the FFT or color depth reduction (feature squeezing).
        predictions = self._Adversarial__model.predictions(adv_transformed)
        assert predictions.ndim == 1
        # The whole inference should be given the initial adversarial (adv)
        # example since the "noisy channel" is a part of the inference. Note
        # that the predictions are given from the network run with transformed
        # adversarial example.
        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
            adv, predictions, in_bounds)


        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial
