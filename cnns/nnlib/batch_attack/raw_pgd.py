import torch
from torch import nn
from typing import Union


def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int],
            eps: float) -> torch.Tensor:
    """Projects x_adv into the l_norm ball around x

    Assumes x and x_adv are 4D Tensors representing batches of images

    Args:
        x: Batch of natural images
        x_adv: Batch of adversarial images
        norm: Norm of ball around x
        eps: Radius of ball

    Returns:
        x_adv: Adversarial examples projected to be at most eps
            distance from x under a certain norm
    """
    if x.shape != x_adv.shape:
        raise ValueError('Input Tensors must have the same shape')

    if norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # Assume x and x_adv are batched tensors where the first dimension is
        # a batch dimension
        mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

        scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
        scaling_factor[mask] = eps

        # .view() assumes batched images as a 4D Tensor
        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

        x_adv = x + delta

    return x_adv


def random_perturbation(x: torch.Tensor,
                        norm: Union[str, int],
                        eps: float) -> torch.Tensor:
    """Applies a random l_norm bounded perturbation to x

    Assumes x is a 4D Tensor representing a batch of images

    Args:
        x: Batch of images
        norm: Norm to measure size of perturbation
        eps: Size of perturbation

    Returns:
        x_perturbed: Randomly perturbed version of x
    """
    perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
    if norm == 'inf':
        perturbation = torch.sign(perturbation) * eps
    else:
        perturbation = project(torch.zeros_like(x), perturbation, norm, eps)

    return x + perturbation


class RAW_PGD:

    def __init__(self,
                 model,
                 num_steps: int = 120,
                 step_size: int = 0.01,
                 eps: float = 2.0,
                 norm=2,
                 clamp=(0, 1),
                 y_target=None,
                 loss_fn=nn.CrossEntropyLoss(),
                 random: bool = True):
        self.model = model
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.norm = norm
        self.eps = eps
        self.clamp = clamp
        self.y_target = y_target
        self.random = random

    def projected_gradient_descent(self, x, y):
        """Performs the projected gradient descent attack on a batch of images."""
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        targeted = self.y_target is not None
        num_channels = x.shape[1]

        if self.random:
            x_adv = random_perturbation(x_adv, self.norm, self.eps)

        for i in range(self.num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = self.model(_x_adv)
            loss = self.loss_fn(prediction, self.y_target if targeted else y)
            loss.backward()

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                if self.norm == 'inf':
                    gradients = _x_adv.grad.sign() * self.step_size
                else:
                    # Note .view() assumes batched image data as 4D tensor
                    gradients = _x_adv.grad * self.step_size / _x_adv.grad.view(
                        _x_adv.shape[0], -1) \
                        .norm(self.norm, dim=-1) \
                        .view(-1, num_channels, 1, 1)

                if targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t. the image data
                    x_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                    # the model parameters
                    x_adv += gradients

            # Project back into l_norm ball and correct range
            if self.norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                x_adv = torch.max(torch.min(x_adv, x + self.eps), x - self.eps)
            else:
                delta = x_adv - x

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(self.norm,
                                                           dim=1) <= self.eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(self.norm,
                                                                     dim=1)
                scaling_factor[mask] = self.eps

                # .view() assumes batched images as a 4D Tensor
                delta *= self.eps / scaling_factor.view(-1, 1, 1, 1)

                x_adv = x + delta

            x_adv = x_adv.clamp(*self.clamp)

        return x_adv.detach()
