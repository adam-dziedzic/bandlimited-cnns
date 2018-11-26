import torch

from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import SchedulerType
from cnns.nnlib.utils.general_utils import LossType
from cnns.nnlib.utils.general_utils import LossReduction
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import Bool
from cnns.nnlib.utils.general_utils import StrideType

class Arguments(object):
    """
    Encapsulate all the arguments for the running program and carry them through
    the execution.
    """

    def __next_counter(self):
        """
        Each variable of an instance of the class is given its unique index.

        :return: return the first unused index.
        """
        self.__counter__ += 1
        return self.__counter__

    def __init__(self,
                 is_debug=False,
                 network_type=NetworkType.ResNet18,
                 # network_type=NetworkType.FCNN_STANDARD,
                 preserve_energy=100,
                 preserved_energies=[100],
                 dtype=torch.float,
                 use_cuda=True,
                 compress_type=CompressType.STANDARD,
                 index_back=0,
                 weight_decay=5e-4,
                 epochs=1,
                 min_batch_size=128,
                 test_batch_size=100,
                 learning_rate=0.1,
                 momentum=0.9,
                 seed=31,
                 log_interval=1,
                 optimizer_type=OptimizerType.MOMENTUM,
                 scheduler_type=SchedulerType.ReduceLROnPlateau,
                 loss_type=LossType.CROSS_ENTROPY,
                 loss_reduction=LossReduction.ELEMENTWISE_MEAN,
                 memory_type=MemoryType.STANDARD,
                 workers=6,
                 model_path="no_model",
                 dataset="cifar10",
                 # dataset="ucr",
                 mem_test=False,
                 is_data_augmentation=True,
                 sample_count_limit=1024,
                 conv_type=ConvType.FFT2D,
                 # conv_type=ConvType.FFT1D,
                 visualize=False,
                 static_loss_scale=1,
                 out_size=None,
                 tensor_type=TensorType.FLOAT32,
                 next_power2=False,
                 dynamic_loss_scale=True,
                 memory_size=25,
                 is_progress_bar=False,
                 stride_type=StrideType.STANDARD,
                 # is_dev_dataset = True,
                 is_dev_dataset=False,
                 dev_percent = 30
                 ):
        """
        The default parameters for the execution of the program.

        :param is_debug: should we print the debug messages.
        :param network_type: the type of network architecture
        :param preserve_energy: how much energy in the input should be preserved
        after compression
        :param dtype: the type of the tensors
        :param use_cuda: should use gpu or not
        :param compress_type: the type of FFT compression, NO_FILTER - do not
        compress the filter. BIG_COEF: preserve only the largest coefficients
        in the frequency domain.
        :param index_back: how much compress based on discarded coefficients
        in the frequency domain.
        :param weight_decay: weight decay for optimizer parameters
        :param epochs: number of epochs for training
        :param min_batch_size: mini batch size for training
        :param test_batch_size: batch size for testing
        :param learning_rate: the optimizer learning rate
        :param momentum: for the optimizer SGD
        :param out_size: users can specify arbitrary output size that we can
        deliver based on spectral pooling
        :param memory_size: specify how much memory can be used (or is available
        on the machine).
        :param is_progress_bar: specify if the progress bar should be shown
        during training and testing of the model.
        :param is_dev_set: is the dev dataset used (extracted from the trina set)
        :param dev_percent: % of data used from the train set as the dev set
        """
        super(Arguments).__init__()
        self.__counter__ = 0

        self.is_debug = is_debug
        self.__idx_is_debug = self.__next_counter()

        self.network_type = network_type
        self.__idx_network_type = self.__next_counter()

        self.preserve_energy = preserve_energy
        self.__idx_preserve_energy = self.__next_counter()

        self.dtype = dtype
        self.__idx_dtype = self.__next_counter()

        if use_cuda is None:
            self.use_cuda = True if torch.cuda.is_available() else False
        else:
            self.use_cuda = use_cuda
        self.__idx_use_cuda = self.__next_counter()

        self.compress_type = compress_type
        self.__idx_compress_type = self.__next_counter()

        self.index_back = index_back
        self.__idx_index_back = self.__next_counter()

        self.weight_decay = weight_decay
        self.__idx_weight_decay = self.__next_counter()

        # Object's variable that are not convertible to an element of a tensor.
        self.preserve_energies = preserved_energies
        self.epochs = epochs
        self.min_batch_size = min_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed
        self.log_interval = log_interval
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.loss_type = loss_type
        self.loss_reduction = loss_reduction
        self.memory_type = memory_type
        self.workers = workers
        self.model_path = model_path
        self.dataset = dataset
        self.mem_test = mem_test
        self.is_data_augmentation = is_data_augmentation
        self.sample_count_limit = sample_count_limit
        self.conv_type = conv_type
        self.visulize = visualize
        self.static_loss_scale = static_loss_scale
        self.out_size = out_size
        self.tensor_type = tensor_type
        self.next_power2 = next_power2
        self.dynamic_loss_scale = dynamic_loss_scale
        self.memory_size = memory_size
        self.is_progress_bar = is_progress_bar
        self.stride_type=stride_type
        self.is_dev_dataset = is_dev_dataset
        self.dev_percent = dev_percent

    def get_bool(self, arg):
        return True if Bool[arg] is Bool.TRUE else False

    def set_parsed_args(self, parsed_args):

        # Make sure you do not miss any properties.
        # https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
        self.__dict__ = parsed_args.__dict__.copy()
        self.parsed_args = parsed_args

        # Enums:
        self.network_type = NetworkType[parsed_args.network_type]
        self.compress_type = CompressType[parsed_args.compress_type]
        self.conv_type = ConvType[parsed_args.conv_type]
        self.optimizer_type = OptimizerType[parsed_args.optimizer_type]
        self.scheduler_type = SchedulerType[parsed_args.scheduler_type]
        self.loss_type = LossType[parsed_args.loss_type]
        self.loss_reduction = LossReduction[parsed_args.loss_reduction]
        self.memory_type = MemoryType[parsed_args.memory_type]
        self.tensor_type = TensorType[parsed_args.tensor_type]
        self.stride_type = StrideType[parsed_args.stride_type]

        # Bools:
        self.is_debug = self.get_bool(parsed_args.is_debug)
        self.dynamic_loss_scale = self.get_bool(parsed_args.dynamic_loss_scale)
        self.next_power2 = self.get_bool(parsed_args.next_power2)
        self.visulize = self.get_bool(parsed_args.visualize)
        self.is_progress_bar = self.get_bool(parsed_args.is_progress_bar)
        self.is_data_augmentation = self.get_bool(parsed_args.is_data_augmentation)
        self.is_dev_dataset = self.get_bool(parsed_args.is_dev_dataset)
        self.mem_test = self.get_bool(parsed_args.mem_test)
        self.use_cuda = self.get_bool(parsed_args.use_cuda) and torch.cuda.is_available()

        if hasattr(parsed_args, "preserve_energy"):
            self.preserve_energy = parsed_args.preserve_energy

    def get_str(self):
        args_dict = self.__dict__
        args_str = ",".join(
            [str(key) + ":" + str(value) for key, value in args_dict.items()])
        return args_str

    def from_bool_arg(self, arg):
        """
        From bool arg to element of a tensor.
        :param arg: bool arg
        :return: int
        """
        return 1 if self.is_debug else -1

    def to_bool_arg(self, arg):
        """
        From element of a tensor to a bool arg.
        :param arg: int
        :return: bool
        """
        if arg == 1:
            return True
        elif arg == -1:
            return False
        else:
            Exception(
                f"Unknown int value for the trarnsformation to bool: {arg}")

    def from_float_arg(self, arg):
        if arg is None:
            return -1
        else:
            return arg

    def to_float_arg(self, arg):
        if arg == -1:
            return None
        else:
            return arg

    def to_tensor(self):
        t = torch.empty(self.__counter__, dtype=torch.float,
                        device=torch.device("cpu"))

        t[self.__idx_network_type] = self.network_type.value
        t[self.__idx_is_debug] = self.from_bool_arg(self.is_debug)
        t[self.__idx_use_cuda] = self.from_bool_arg(self.use_cuda)
        t[self.__idx_compress_type] = self.compress_type.value
        t[self.__idx_preserve_energy] = self.from_float_arg(
            self.preserve_energy)
        t[self.__idx_index_back] = self.from_float_arg(self.index_back)

    def from_tensor(self, t):
        self.network_type = NetworkType(t[int(self.__idx_nework_type)])
        self.is_debug = self.to_bool_arg(t[self.__idx_is_debug])
        self.use_cuda = self.to_bool_arg(t[self.__idx_use_cuda])
        self.compress_type = CompressType(t[int(self.__idx_compress_type)])
        self.preserve_energy = self.to_float_arg(t[self.__idx_preserve_energy])
        self.index_back = self.to_float_arg(t[self.__idx_index_back])