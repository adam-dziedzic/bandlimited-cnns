from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.utils import softmax
from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack

import numpy as np
nprng = np.random.RandomState()

def sample_noise(epsilon, bounds, shape, dtype):
    """
    Simiarl to foolbox but batched version.
    :param epsilon: strength of the noise
    :param bounds: min max for images
    :param shape: the output shape
    :param dtype: the output type
    :return: the noise for images
    """
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = nprng.uniform(-w, w, size=shape)
    noise = noise.astype(dtype)
    return noise

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
    result.L1_distance = 0
    result.L2_distance = 0
    result.Linf_distance = 0
    avg_predictions = np.array([0.0] * args.num_classes)
    class_id_counters = [0] * args.num_classes
    batch_size = args.test_batch_size
    C, H, W = image.shape

    if is_batch:
        iters = iters // batch_size + 1
    else:
        batch_size = 1

    for iter in range(iters):
        noise = sample_noise(
            epsilon=args.noise_epsilon,
            shape=(batch_size, C, H, W),
            dtype=image.dtype,
            bounds=(args.min, args.max))
        noise_images = image + noise
        noise_image = np.average(noise_images, axis=0)
        predictions = fmodel.batch_predictions(images=noise_images)
        predictions = np.average(predictions, axis=0)
        soft_predictions = softmax(predictions)
        predicted_class_id = np.argmax(soft_predictions)

        avg_predictions += predictions
        class_id_counters[predicted_class_id] += 1
        result.confidence += np.max(soft_predictions)

        result.L2_distance += meter.measure(image, noise_image, norm=2)
        result.L1_distance += meter.measure(image, noise_image, norm=1)
        result.Linf_distance += meter.measure(image, noise_image,
                                              norm=float('inf'))

    result.class_id = np.argmax(np.array(class_id_counters))
    result.label = from_class_idx_to_label[result.class_id]

    avg_predictions /= iters

    result.confidence /= iters
    # result.L2_distance /= iters
    result.L1_distance /= iters
    result.Linf_distance /= iters

    return result, avg_predictions
