import time
import numpy as np
from cnns.nnlib.utils.exec_args import get_args
import cnns.foolbox.foolbox_2_3_0 as foolbox
from cnns.nnlib.robustness.pytorch_model import get_model
from cnns.nnlib.datasets.load_data import get_data
from cnns.nnlib.robustness.param_perturbation.utils import get_adv_images


def compute(args):
    pytorch_model = get_model(args)
    preprocessing = dict(mean=args.mean_array,
                         std=args.std_array,
                         axis=-3)
    fmodel = foolbox.models.PyTorchModel(pytorch_model,
                                         bounds=(args.min, args.max),
                                         channel_axis=1,
                                         device=args.device,
                                         num_classes=args.num_classes,
                                         preprocessing=(0, 1))

    train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
        args=args)

    if args.target_class > -1:
        criterion = foolbox.criteria.TargetClass(target_class=args.target_class)
        print(f'target class id: {args.target_class}')
        print(
            f'target class name: {args.from_class_idx_to_label[args.target_class]}')
    else:
        criterion = foolbox.criteria.Misclassification()
        print('No target class specified')

    attack = foolbox.attacks.CarliniWagnerL2Attack(
        fmodel, criterion=criterion)

    total_count = 0
    clean_count = 0
    adv_count = 0
    sum_distances = 0
    attacks_failed = 0

    if args.use_set == 'test_set':
        data_loader = test_loader
    elif args.use_set == 'train_set':
        data_loader = train_loader
    else:
        raise Exception("Unknown use set: ", args.use_set)
    print(f"Using set: {args.use_set} for source of data to find "
          f"adversarial examples.")

    for batch_idx, (images, labels) in enumerate(data_loader):
        total_count += len(labels)
        images, labels = images.numpy(), labels.numpy()
        clean_labels = fmodel.forward(images).argmax(axis=-1)

        clean_count += np.sum(clean_labels == labels)
        print('clean accuracy: ', clean_count / total_count)

        adversarials = attack(images, labels, unpack=False,
                              max_iterations=args.attack_max_iterations,
                              binary_search_steps=args.binary_search_steps,
                              initial_const=args.attack_strength,
                              confidence=args.attack_confidence
                              )

        adversarial_classes = np.asarray(
            [a.adversarial_class for a in adversarials])
        # print('orginal labels: ', labels)
        # print('adversarial labels: ', adversarial_classes)
        # print('count how many original and adversarial classes agree: ',
        #       np.sum(adversarial_classes == labels))  # will always be 0.0
        adv_count += np.sum(adversarial_classes == labels)

        # The `Adversarial` objects also provide a `distance` attribute.
        # Note that the distances
        # can be 0 (misclassified without perturbation) and inf (attack failed).
        distances = np.asarray([a.distance.value for a in adversarials])
        distances = np.asarray([distance for distance in distances if
                                distance != 0.0 and distance != np.inf])
        sum_distances += np.sum(distances)
        print('avg distance: ', sum_distances / total_count)

        attacks_failed += sum(
            adv.distance.value == np.inf for adv in adversarials)
        print("{} of {} attacks failed".format(attacks_failed, total_count))

        print("{} of {} inputs misclassified without perturbation".format(
            sum(adv.distance.value == 0 for adv in adversarials),
            len(adversarials)))

        advs = get_adv_images(adversarials=adversarials, images=images)
        adv_acc = np.mean(fmodel.forward(advs).argmax(axis=-1) == labels)
        print('adversarial accuracy: ', adv_acc)

    print('total count: ', total_count)
    if total_count > 0:
        print('avg distance: ', sum_distances / total_count)
        print('clean accuracy: ', clean_count / total_count)
        print('attacks failed: ', attacks_failed / total_count)
        print('adv accuracy: ', adv_count / total_count)


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    for attack_iterations in args.many_attack_iterations:
        args.attack_max_iterations = attack_iterations
        for attack_strength in args.attack_strengths:
            args.attack_strength = attack_strength

    compute(args)
    print("total elapsed time: ", time.time() - start_time)
