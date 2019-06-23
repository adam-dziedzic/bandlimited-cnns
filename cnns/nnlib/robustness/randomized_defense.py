from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.utils import softmax
from cnns.nnlib.robustness.utils import uniform_noise
from cnns.nnlib.robustness.utils import laplace_noise
from cnns.nnlib.robustness.utils import gauss_noise
from cnns.nnlib.robustness.utils import elem_wise_dist
from cnns.nnlib.robustness.utils import most_frequent_class
import numpy as np
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
nprng = np.random.RandomState()


def defend(image, fmodel, args, iters=None, is_batch=True, original_image=None):
    """
    Recover the correct label.

    :param image: the input image (after attack)
    :param fmodel: a foolbox model
    :param args: the global arguments
    :param original_image: the original image

    :return: the result object with selected label, distances and confidence
    """
    if iters is None:
        iters = args.noise_iterations
    from_class_idx_to_label = args.from_class_idx_to_label

    result = Object()
    result.confidence = 0
    if original_image is not None:
        result.L1_distance = []
        result.L2_distance = []
        result.Linf_distance = []
    else:
        result.L1_distance = [-1]
        result.L2_distance = [-1]
        result.Linf_distance = [-1]

    avg_predictions = np.array([0.0] * args.num_classes)
    avg_confidence = np.array([0.0] * args.num_classes)
    class_id_counters = [0] * args.num_classes
    batch_size = args.test_batch_size
    C, H, W = image.shape

    # Do the summation / norm measurement for the "inner" images, omit the
    # batch dimension: 0
    axis = (1, 2, 3)

    if is_batch:
        if iters <= batch_size:
            batch_size = iters
            iters = 1
        else:
            iters = iters // batch_size
            if iters % batch_size != 0:
                iters += 1
    else:
        batch_size = 1

    if args.noise_epsilon > 0:
        epsilon = args.noise_epsilon
        noiser = uniform_noise
    elif args.laplace_epsilon > 0:
        noiser = laplace_noise
        epsilon = args.laplace_epsilon
    elif args.sigma_epsilon > 0:
        noiser = gauss_noise
        epsilon = args.noise_sigma
    else:
        raise Exception("No noise was used.")

    for iter in range(iters):
        noise = noiser(
            epsilon=epsilon,
            shape=(batch_size, C, H, W),
            dtype=image.dtype,
            args=args)
        noise_images = image + noise
        predictions = fmodel.batch_predictions(images=noise_images)
        # Aggregate predictions via average.
        agg_predictions = np.average(predictions, axis=0)
        assert len(agg_predictions) == args.num_classes
        soft_predictions = softmax(agg_predictions)

        # We use the pluralism rule - find the most frequent class.
        predicted_class_id = most_frequent_class(predictions)

        # Alternatively, we could compute the final class from the soft
        # predictions.
        # predicted_class_id = np.argmax(soft_predictions)

        avg_predictions += agg_predictions
        avg_confidence += soft_predictions
        class_id_counters[predicted_class_id] += 1

        batch_mode = True
        if original_image is not None:
            if batch_mode:
                denorm_original_image = args.denormalizer.denormalize(
                    original_image)
                denormalizer_noise_images = Denormalize(
                    mean_array=np.expand_dims(args.mean_array, axis=0),
                    std_array=np.expand_dims(args.std_array, axis=0))
                denorm_noise_images = denormalizer_noise_images.denormalize(
                    noise_images)
                result.L2_distance = np.append(
                result.L2_distance, elem_wise_dist(
                    denorm_original_image, denorm_noise_images, p=2, axis=axis))
                # print('result L2 distance batch: ', result.L2_distance)
                result.L1_distance = np.append(
                    result.L1_distance, elem_wise_dist(
                        denorm_original_image, denorm_noise_images, p=1,
                        axis=axis))
                result.Linf_distance = np.append(
                    result.Linf_distance, elem_wise_dist(
                        denorm_original_image, denorm_noise_images,
                        p=float('inf'), axis=axis))
            else:
                for noise_image in noise_images:
                    L2 = args.meter.measure(original_image, noise_image, norm=2)
                    result.L2_distance.append(L2)
                    L1 = args.meter.measure(original_image, noise_image, norm=1)
                    result.L1_distance.append(L1)
                    Linf = args.meter.measure(original_image, noise_image,
                                              norm=float('inf'))
                    result.Linf_distance.append(Linf)
                # print('result L2 distance iterative: ', result.L2_distance)
        result.class_id = np.argmax(np.array(class_id_counters))
    result.label = from_class_idx_to_label[result.class_id]

    avg_predictions /= iters
    result.confidence = np.max(avg_confidence / iters)

    result.L2_distance = np.average(result.L2_distance)
    result.L1_distance = np.average(result.L1_distance)
    result.Linf_distance = np.average(result.Linf_distance)

    return result, avg_predictions, class_id_counters
