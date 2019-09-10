from cnns.nnlib.utils.general_utils import LossType
from cnns.nnlib.pytorch_experiments.utils.optim_utils import get_optimizer
from cnns.nnlib.pytorch_experiments.utils.optim_utils import get_loss_function
import torch
import numpy as np
import torch.nn.functional as F


def get_default_loss_function(loss_type=LossType.CROSS_ENTROPY,
                              reduction_function="mean"):
    """
    :param loss: "CROSS_ENTROPY", "NLL", or "MSE"
    :param reduction_function: "mean" or "sum"
    :return: the loss function
    """
    if loss_type is LossType.CROSS_ENTROPY:
        loss_function = torch.nn.CrossEntropyLoss(reduction=reduction_function)
    elif loss_type is LossType.NLL:
        loss_function = torch.nn.NLLLoss(reduction=reduction_function)
    elif loss_type is LossType.MSE:
        loss_function = torch.nn.MSELoss(reduction=reduction_function)
    else:
        raise Exception(f"Unknown loss type: {loss_type.name}")
    return loss_function


def get_gradient_for_input(args, model, input: torch.tensor, target,
                           loss_function=torch.nn.CrossEntropyLoss()):
    if input is None:
        return None
    if input.ndim == 3:
        input = np.expand_dims(input, axis=0)
        target = np.expand_dims(target, axis=0)
    data = torch.tensor(input, device=args.device, requires_grad=True)
    target = torch.tensor(target, device=args.device)

    model.to(args.device)
    output = model(data)
    predicted_class = torch.argmax(output).item()
    confidence = torch.max(F.softmax(output)).item()
    loss = loss_function(output, target)
    loss.backward()
    loss_value = loss.item()
    gradient = data.grad.detach().cpu().numpy()
    # l2_norm = args.meter.measure_single_numpy(gradient)
    # l2_norm = np.sqrt(np.sum(gradient * gradient))
    l2_norm = np.sum(gradient * gradient)
    return gradient, loss_value, predicted_class, l2_norm, np.min(
        gradient), np.mean(gradient), np.max(gradient), confidence


def compute_gradients(args, model, original_image: torch.tensor, original_label,
                      adv_image: torch.tensor, adv_label, gauss_image=None):
    grads = {}
    target = 1
    grad_original_correct = get_gradient_for_input(args=args, model=model,
                                                   input=original_image,
                                                   target=original_label)
    assert grad_original_correct[2] == original_label, 'wrong classification'
    grad_original_adv = get_gradient_for_input(args=args, model=model,
                                               input=original_image,
                                               target=adv_label)
    grad_original_zero = get_gradient_for_input(args=args, model=model,
                                                input=original_image,
                                                target=target)
    assert grad_original_adv[2] == original_label, 'wrong classification'
    grad_adv_correct = get_gradient_for_input(args=args, model=model,
                                              input=adv_image,
                                              target=original_label)
    assert grad_adv_correct[2] == adv_label, 'wrong classification'
    grad_adv_adv = get_gradient_for_input(args=args, model=model,
                                          input=adv_image,
                                          target=adv_label)
    assert grad_adv_adv[2] == adv_label, 'wrong classification'
    grad_adv_zero = get_gradient_for_input(args=args, model=model,
                                           input=adv_image,
                                           target=target)
    grad_gauss_correct = get_gradient_for_input(args=args, model=model,
                                                input=gauss_image,
                                                target=original_label)
    grad_gauss_adv = get_gradient_for_input(args=args, model=model,
                                            input=gauss_image, target=adv_label)
    grad_gauss_zero = get_gradient_for_input(args=args, model=model,
                                             input=gauss_image, target=target)
    grads['original_correct'] = grad_original_correct
    grads['original_adv'] = grad_original_adv
    grads['original_zero'] = grad_original_zero

    grads['adv_correct'] = grad_adv_correct
    grads['adv_adv'] = grad_adv_adv
    grads['adv_zero'] = grad_adv_zero

    grads['gauss_correct'] = grad_gauss_correct
    grads['gauss_adv'] = grad_gauss_adv
    grads['gauss_zero'] = grad_gauss_zero

    results = dict()
    results['l2_norm_adv_adv'] = np.sqrt(np.sum(grad_adv_adv[0] * grad_adv_adv[0]))
    results['l2_norm_adv_correct'] = np.sqrt(np.sum(grad_adv_correct[0] * grad_adv_correct[0]))
    results['l2_norm_adv_zero'] = np.sqrt(np.sum(grad_adv_zero[0] * grad_adv_zero[0]))

    results['adv_dot_adv_adv'] = np.sum(grad_adv_adv[0] * grad_adv_adv[0])
    results['adv_dot_adv_correct'] = np.sum(grad_adv_adv[0] * grad_adv_correct[0])
    results['adv_dot_adv_zero'] = np.sum(grad_adv_adv[0] * grad_adv_zero[0])
    results['adv_dot_correct_zero'] = np.sum(grad_adv_correct[0] * grad_adv_zero[0])

    results['l2_norm_original_adv'] = np.sqrt(np.sum(grad_original_adv[0] * grad_original_adv[0]))
    results['l2_norm_original_correct'] = np.sqrt(np.sum(grad_original_correct[0] * grad_original_correct[0]))
    results['l2_norm_original_zero'] = np.sqrt(np.sum(grad_original_zero[0] * grad_original_zero[0]))

    results['original_dot_correct_correct'] = np.sum(grad_original_correct[0] * grad_original_correct[0])
    results['original_dot_adv_adv'] = np.sum(grad_original_adv[0] * grad_original_adv[0])
    results['original_dot_zero_zero'] = np.sum(grad_original_zero[0] * grad_original_zero[0])
    results['original_dot_adv_correct'] = np.sum(grad_original_adv[0] * grad_original_correct[0])
    results['original_dot_adv_zero'] = np.sum(grad_original_adv[0] * grad_original_zero[0])
    results['original_dot_correct_zero'] = np.sum(grad_original_correct[0] * grad_original_zero[0])

    results['l2_norm_gauss_adv'] = np.sqrt(np.sum(grad_gauss_adv[0] * grad_gauss_adv[0]))
    results['l2_norm_gauss_correct'] = np.sqrt(np.sum(grad_gauss_correct[0] * grad_gauss_correct[0]))
    results['l2_norm_gauss_zer'] = np.sqrt(np.sum(grad_gauss_zero[0] * grad_gauss_zero[0]))

    results['gauss_dot_correct_correct'] = np.sum(grad_gauss_correct[0] * grad_gauss_correct[0])
    results['gauss_dot_adv_correct'] = np.sum(grad_gauss_adv[0] * grad_gauss_correct[0])
    results['gauss_dot_adv_zero'] = np.sum(grad_gauss_adv[0] * grad_gauss_zero[0])
    results['gauss_dot_correct_zero'] = np.sum(grad_gauss_correct[0] * grad_gauss_zero[0])

    for grad_key in sorted(grads.keys()):
        grad = grads[grad_key]
        info = [grad_key, ' grad L2: ', grad[3], ' loss value', grad[1],
                ' predicated class', grad[2], 'min', grad[4], 'mean', grad[5],
                'max', grad[6], 'confidence', grad[7]]
        print(",".join([str(x) for x in info]))

    return grads, results
