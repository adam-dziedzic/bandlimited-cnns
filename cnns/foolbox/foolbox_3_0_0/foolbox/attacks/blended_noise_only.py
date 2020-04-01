from typing import Union, Optional, Any
import eagerpy as ep

from ..devutils import atleast_kd

from ..distances import Distance

from .base import FlexibleDistanceMinimizationAttack
from .base import Model
from .base import Criterion
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import raise_if_kwargs


class BlendedUniformNoiseOnlyAttack(FlexibleDistanceMinimizationAttack):
    """Blends the input with a uniform noise.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        directions : Number of random directions in which the perturbation is searched.
    """

    def __init__(
            self,
            *,
            distance: Optional[Distance] = None,
            directions: int = 1000,
    ):
        super().__init__(distance=distance)
        self.directions = directions
        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, Any] = None,
            *,
            early_stop: Optional[float] = None,
            **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        is_adversarial = get_is_adversarial(criterion_, model)

        min_, max_ = model.bounds

        for j in range(self.directions):
            # random noise inputs tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_ = ep.uniform(x, x.shape, min_, max_)
            is_adv_ = atleast_kd(is_adversarial(random_), x.ndim)

            if j == 0:
                random = random_
                is_adv = is_adv_
            else:
                random = ep.where(is_adv, random, random_)
                is_adv = is_adv.logical_or(is_adv_)

            if is_adv.all():
                break

        return random
