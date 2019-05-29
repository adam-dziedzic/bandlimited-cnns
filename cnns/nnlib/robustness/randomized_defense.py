from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.utils import softmax
from cnns.nnlib.robustness.utils import sample_noise
from cnns.nnlib.robustness.utils import elem_wise_dist
# from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack

import numpy as np

nprng = np.random.RandomState()


def defend(image, fmodel, args, iters=None, is_batch=True):
    """
    Recover the correct label.

    :param image: the input image (after attack)
    :param fmodel: a foolbox model
    :param args: the global arguments
    :return: the result object with selected label, distances and confidence
    """
    if iters is None:
        iters = args.noise_iterations
    meter = DenormDistance(mean_array=args.mean_array, std_array=args.std_array)
    from_class_idx_to_label = args.from_class_idx_to_label

    result = Object()
    result.confidence = 0
    result.L1_distance = []
    result.L2_distance = []
    result.Linf_distance = []
    avg_predictions = np.array([0.0] * args.num_classes)
    avg_confidence = np.array([0.0] * args.num_classes)
    class_id_counters = [0] * args.num_classes
    batch_size = args.test_batch_size
    C, H, W = image.shape

    # Do the summation / norm measurement for the "inner" images, omit the
    # batch dimension: 0
    dims = (1, 2, 3)

    if is_batch:
        iters = iters // batch_size
        if iters == 0:
            iters = 1
    else:
        batch_size = 1

    for iter in range(iters):
        noise = sample_noise(
            epsilon=args.noise_epsilon,
            shape=(batch_size, C, H, W),
            dtype=image.dtype,
            bounds=(args.min, args.max))
        noise_images = image + noise
        predictions = fmodel.batch_predictions(images=noise_images)
        predictions = np.average(predictions, axis=0)
        assert len(predictions) == args.num_classes
        soft_predictions = softmax(predictions)
        predicted_class_id = np.argmax(soft_predictions)

        avg_predictions += predictions
        avg_confidence += soft_predictions
        class_id_counters[predicted_class_id] += 1

        result.L2_distance += elem_wise_dist(image, noise_images,
                                             p=2, dim=dims)
        result.L1_distance += elem_wise_dist(image, noise_images,
                                             p=1, dim=dims)
        result.Linf_distance += elem_wise_dist(image, noise_images,
                                               p=float('inf'), dim=dims)

    result.class_id = np.argmax(np.array(class_id_counters))
    result.label = from_class_idx_to_label[result.class_id]

    avg_predictions /= iters
    result.confidence = avg_confidence / iters

    result.L2_distance = np.average(result.L2_distance)
    result.L1_distance = np.average(result.L1_distance)
    result.Linf_distance = np.average(result.Linf_distance)

    return result, avg_predictions, class_id_counters
