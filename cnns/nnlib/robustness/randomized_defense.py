from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.utils import softmax
from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack

import numpy as np

def defend(image, fmodel, args):
    """
    Recover the correct label.

    :param image: the input image (after attack)
    :param fmodel: a foolbox model
    :param args: the global arguments
    :return: the result object with selected label, distances and confidence
    """
    iters = args.noise_iterations
    meter = DenormDistance(mean_array=args.mean_array,
                           std_array=args.std_array)
    from_class_idx_to_label = args.from_class_idx_to_label

    result = Object()
    result.confidence = 0
    result.L1_distance = 0
    result.L2_distance = 0
    result.Linf_distance = 0
    result.avg_predictions = np.array([0.0] * args.num_classes)
    class_id_counters = [0] * args.num_classes

    noiser = AdditiveUniformNoiseAttack()
    for iter in range(iters):
        noise = noiser._sample_noise(
            epsilon=args.noise_epsilon, image=image,
            bounds=(args.min, args.max))
        noise_image = image + noise
        predictions = fmodel.predictions(image)
        result.avg_predictions += predictions
        soft_predictions = softmax(predictions)
        predicted_class_id = np.argmax(soft_predictions)
        class_id_counters[predicted_class_id] += 1
        result.confidence += np.max(soft_predictions)
        result.L2_distance += meter.measure(image, noise_image)
        result.L1_distance += meter.measure(image, noise_image, norm=1)
        result.Linf_distance += meter.measure(image, noise_image,
                                             norm=float('inf'))

    max_counter = 0
    max_class_id = 0
    for class_id, class_counter in enumerate(class_id_counters):
        if class_counter > max_counter:
            max_class_id = class_id

    result.class_id = max_class_id
    result.label = from_class_idx_to_label[max_class_id]

    result.avg_predictions /= iters

    result.confidence /= iters
    result.L1_distance /= iters
    result.L2_distance /= iters
    result.Linf_distance /= iters

    return result