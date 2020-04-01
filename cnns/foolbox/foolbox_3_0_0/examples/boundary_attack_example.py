#!/usr/bin/env python3
import torchvision.models as models
import torch
import eagerpy as ep
from cnns.foolbox.foolbox_3_0_0 import foolbox
import time

if __name__ == "__main__":
    # instantiate a model
    model = models.resnet50(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    batchsize = 20
    dataset = 'imagenet'
    # dataset = 'cifar10'
    # images, labels = ep.astensors(*foolbox.samples(fmodel, dataset=dataset, batchsize=batchsize))
    images, labels = foolbox.samples(fmodel, dataset=dataset, batchsize=batchsize)
    print('clean accuracy: ', foolbox.accuracy(fmodel, images, labels))

    a = torch.arange(9, dtype=torch.float).reshape((3, 3))
    print('images size: ', images.size())
    p = 0
    print('norm ', str(p), torch.norm(a, p=p, dim=(1)))
    print('norm ', str(p), torch.norm(images, p=p, dim=(1, 2, 3)))

    print('original labels: ', labels)
    predictions_logits = fmodel(images)
    predictions = torch.argmax(predictions_logits, dim=-1)
    print('original predictions: ', predictions)
    print('original accuracy: ', labels.eq(predictions).float().mean().item())
    # apply the attack
    # attack = foolbox.attacks.LinfPGD()

    # steps_list = [0, 10, 100, 1000, 10000, 25000, 100000]
    steps_list = [100]
    # epsilons = [1.0, 0.5, 0.3, 0.1, 0.03]
    epsilons = None

    # calculate and report the robust accuracy
    for steps in steps_list:
        start = time.time()
        attack = foolbox.attacks.BoundaryAttack(steps=steps)
        adv_images, _, success = attack(fmodel, images, labels, epsilons=epsilons)
        # print('adversarial labels: ', torch.argmax(model(images), dim=1))
        robust_accuracy = 1 - success.float().mean(axis=-1)
        print('steps: ', steps)
        for eps, acc, advs in zip(epsilons, robust_accuracy, adv_images):
            print('eps: ', eps, ' robust accuracy: ', acc.item())

            adv_logits = fmodel(advs)
            adv_predictions = torch.argmax(adv_logits, dim=-1)
            print('adv_predictions: ', adv_predictions)
            print('adv accuracy: ', labels.eq(adv_predictions).float().mean().item())

            # and check if they are smaller than eps
            # print('linf dist: ', (advs - images).norms.linf(axis=(1, 2, 3)).numpy())
            # print('l2 dist: ', (advs - images).norms.l2(axis=(1, 2, 3)).numpy())
            for p in [0, 1, 2, float('inf')]:
                print('norm ', str(p), torch.norm(advs - images, p=p, dim=(1, 2, 3)).mean().item())
                print('norm ', str(p), torch.norm(advs - images, p=p, dim=(1, 2, 3)).max().item())

        stop = time.time()
        elapsed_time = stop - start
        print('total time: ', elapsed_time)

    # we can also manually check this
    # for eps, advs_ in zip(epsilons, advs):
    #    print('espilon: ', eps, 'accuracy: ', foolbox.accuracy(fmodel, advs_, labels))
    #    # but then we also need to look at the perturbation sizes
    #    # and check if they are smaller than eps
    #    print('linf dist: ', (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy())
    #    print('l2 dist: ', (advs_ - images).norms.l2(axis=(1,2,3)).numpy())
