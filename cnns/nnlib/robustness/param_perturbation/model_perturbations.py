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

    total_count = 0
    clean_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        total_count += len(labels)
        images, labels = images.numpy(), labels.numpy()
        clean_labels = fmodel.forward(images).argmax(axis=-1)

        clean_count += np.sum(clean_labels == labels)
        print('clean accuracy: ', clean_count / total_count)

        params = fmodel._model.parameters()



if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    compute(args)
    print("total elapsed time: ", time.time() - start_time)
