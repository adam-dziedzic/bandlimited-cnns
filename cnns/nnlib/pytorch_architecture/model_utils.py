from cnns.nnlib.pytorch_architecture.le_net import LeNet
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.pytorch_architecture.densenet import densenet_cifar
from cnns.nnlib.pytorch_architecture.fcnn import FCNNPytorch
from cnns.nnlib.utils.general_utils import NetworkType

def getModelPyTorch(args):
    """
    Get the PyTorch version of the FCNN model.
    :param input_size: the length (width) of the time series.
    :param num_classes: number of output classes.
    :param in_channels: number of channels in the input data for a convolution.
    :param out_channels: number of channels in the output of a convolution.
    :param dtype: global - the type of torch data/weights.
    :param flat_size: the size of the flat vector after the conv layers.
    :return: the model.
    """
    network_type = args.network_type
    if network_type is NetworkType.LE_NET:
        return LeNet(args=args)
    elif network_type is NetworkType.FCNN_SMALL or (
            network_type is NetworkType.FCNN_STANDARD):
        if network_type is NetworkType.FCNN_SMALL:
            args.out_channels = [1, 1, 1]
        elif network_type is NetworkType.FCNN_STANDARD:
            args.out_channels = [128, 256, 128]
        return FCNNPytorch(args=args)
    elif network_type == NetworkType.ResNet18:
        return resnet18(args=args)
    elif network_type == NetworkType.DenseNetCifar:
        return densenet_cifar(args=args)
    else:
        raise Exception("Unknown network_type: ", network_type)