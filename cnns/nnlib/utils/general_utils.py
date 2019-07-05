import time
from datetime import datetime
import os
import pathlib
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pickle
from enum import Enum

counter = 0

plots_folder_name = "fft_visualize"
plots_dir = os.path.join(os.curdir, plots_folder_name)
pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)


def get_log_time():
    # return time.strftime("%Y-%m-%d-%H-%M-%S", datetime.now())
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


additional_log_file = "additional-info-" + get_log_time() + ".log"
mem_log_file = "mem_log-" + get_log_time() + ".log"


class EnumWithNames(Enum):
    """
    The Enum classes that inherit from the EnumWithNames will get the get_names
    method to return an array of strings representing all possible enum values.
    """

    @classmethod
    def get_names(cls):
        return [enum_value.name for enum_value in cls]


class Bool(EnumWithNames):
    """
    Bool version of EnumWithNames.
    """
    TRUE = 1
    FALSE = 2

    @classmethod
    def get_bool(cls, bool_name):
        bool_enum = cls[bool_name]
        bool_value = True if bool_enum is cls.TRUE else False
        return bool_value


class OptimizerType(EnumWithNames):
    MOMENTUM = 1
    ADAM = 2
    SGD = 3
    ADAM_FLOAT16 = 4


class SchedulerType(EnumWithNames):
    ReduceLROnPlateau = 1
    CosineAnnealingLR = 2
    ExponentialLR = 3
    MultiStepLR = 4
    StepLR = 5
    LambdaLR = 6
    Custom = 7


class LossType(EnumWithNames):
    NLL = 1  # Negative Log Likelihood
    CROSS_ENTROPY = 2


class LossReduction(EnumWithNames):
    SUM = 1
    ELEMENTWISE_MEAN = 2


class MemoryType(EnumWithNames):
    STANDARD = 1
    PINNED = 2


class StrideType(EnumWithNames):
    STANDARD = 1
    SPECTRAL = 2


class RunType(EnumWithNames):
    TEST = 0
    DEBUG = 1


class ModelType(EnumWithNames):
    RES_NET = 0
    DENSE_NET = 1


class PrecisionType(EnumWithNames):
    AMP = 0  # Automatic Mixed Precision
    FP16 = 1
    FP32 = 2


DEFAULT_OPTIMIZER = OptimizerType.ADAM
DEFAULT_RUN_TYPE = RunType.TEST
DEFAULT_MODEL_TYPE = ModelType.DENSE_NET


class ConvType(EnumWithNames):
    STANDARD = 1
    SPECTRAL_PARAM = 2
    SPECTRAL_DIRECT = 3
    SPATIAL_PARAM = 4
    FFT1D = 5
    AUTOGRAD = 6
    SIMPLE_FFT = 7
    SIMPLE_FFT_FOR_LOOP = 8
    COMPRESS_INPUT_ONLY = 9
    FFT2D = 10
    STANDARD2D = 11
    AUTOGRAD2D = 12


class AttackType(EnumWithNames):
    RECOVERY = 1
    ROUND_ONLY = 3
    BAND_ONLY = 2
    ROUND_BAND = 4
    NOISE_ONLY = 5
    NO_ATTACK = 6
    GAUSS_ONLY = 7
    LAPLACE_ONLY = 8
    FFT_RECOVERY = 9
    ROUND_RECOVERY = 10
    SVD_RECOVERY = 11
    GAUSS_RECOVERY = 12
    UNIFORM_RECOVERY = 13
    LAPLACE_RECOVERY = 14


class AdversarialType(EnumWithNames):
    NONE = 1
    BEFORE = 2
    AFTER = 3


class ConvExecType(EnumWithNames):
    SERIAL = 1
    BATCH = 2
    CUDA = 3
    CUDA_SHARED_LOG = 4
    CUDA_DEEP = 5
    SGEMM = 6


class TensorType(EnumWithNames):
    FLOAT32 = 1
    DOUBLE = 2
    FLOAT16 = 3
    INT = 4


class NetworkType(EnumWithNames):
    FCNN_STANDARD = 1
    FCNN_SMALL = 2
    LE_NET = 3
    ResNet18 = 4
    DenseNetCifar = 5  # DenseNet-121 with growth-rate 12
    ResNet50 = 6  # for ImageNet
    Net = 7  # for MNIST


class CompressType(EnumWithNames):
    """

    >>> compress_type = CompressType.BIG_COEFF
    >>> assert compress_type.value == 2
    >>> compress_type = CompressType(3)
    >>> assert compress_type is CompressType.LOW_COEFF
    """
    # Compress both the input signal and the filter.
    STANDARD = 1
    # Preserve the largest (biggest) coefficients and zero-out the lowest
    # (smallest) coefficients.
    BIG_COEFF = 2
    # Preserve only the lowest coefficients but zero-out the highest
    # coefficients.
    LOW_COEFF = 3
    # Compress the filters for fft based convolution or only the input signals.
    NO_FILTER = 4


def energy(x):
    """
    Calculate the energy of the signal.

    :param x: the input signal
    :return: energy of x
    """
    return np.sum(np.power(x, 2))


def reshape_3d_rest(x):
    """
    Reshape the one dimensional input x into a 3 dimensions, with the 1 for each of the first 2 dimensions and the last
    dimension with all the elements of x.

    :param x: a 1D input array
    :return: reshaped array to 3 dimensions with the last one having all the input values
    """
    return x.reshape(1, 1, -1)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def abs_error(x, y):
    """ returns the absolute error """
    return np.sum(np.abs(x - y))


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output)


def plot_signal_raw(signal, title="signal", xlabel=""):
    # matplotlib.use("TkAgg")
    # matplotlib.use("agg")
    matplotlib.use("Qt5Agg")
    plt.plot(range(0, len(signal)), signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    # %matplotlib inline
    # https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
    # Change the backend to display the graphs.
    """
    Resolving this matplotlib  issue involves manually installing dependencies 
    via apt-get  and adjusting the matplotlib backend to use TkAgg , followed by 
    compiling and installing matplotlib  from source. Afterwards, the issue 
    seems to be resolved.
    pip uninstall matplotlib
    sudo apt-get install python-matplotlib
    sudo apt-get install tcl-dev tk-dev python-tk python3-tk
    git clone https://github.com/matplotlib/matplotlib.git
    cd matplotlib
    python setup.py install
    """
    # plt.show()
    global counter
    counter += 1
    plt.savefig(os.path.join(plots_dir,
                             get_log_time() + "-" + title + "-counter-" + str(
                                 counter) + ".png"))
    plt.close()


def plot_signal_time(signal, title="signal", xlabel="Time"):
    pass
    # plot_signal_raw(signal, title, xlabel)


def plot_signal_freq(signal, title="signal", xlabel="Frequency"):
    pass
    # plot_signal_raw(signal, title, xlabel)


def plot_signals(x, y, title="", xlabel="Time", ylabel="Amplitude",
                 label_x="input", label_y="output",
                 linestyle="solid"):
    fontsize = 20
    linewidth = 2.0
    plt.plot(range(0, len(x)), x, color="red", linewidth=linewidth,
             linestyle=linestyle)
    plt.plot(range(0, len(y)), y, color="blue", linewidth=linewidth,
             linestyle=linestyle)
    # We prepare the plot
    fig = plt.figure(1)
    plot = fig.add_subplot(111)
    # We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=fontsize)
    plot.tick_params(axis='both', which='minor', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    red_patch = mpatches.Patch(color='red', label=label_x)
    blue_patch = mpatches.Patch(color='blue', label=label_y)
    plt.legend(handles=[red_patch, blue_patch], fontsize=fontsize,
               loc='upper left')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
    plt.close()


if __name__ == "__main__":
    np_dout = np.array(
        [-9.22318304e-06, -9.19182548e-06, -9.16752651e-06, -9.15080727e-06
            , -9.14208704e-06, -9.14167686e-06, -9.14976863e-06, -9.16643057e-06
            , -9.19160811e-06, -9.22511481e-06, -9.26663870e-06, -9.31573868e-06
            , -9.37185268e-06, -9.43430132e-06, -9.50229423e-06, -9.57493921e-06
            , -9.65125764e-06, -9.73018814e-06, -9.81060839e-06, -9.89134605e-06
            , -9.97119514e-06, -1.00489324e-05, -1.01233354e-05, -1.01931992e-05
            , -1.02573576e-05, 2.13539874e-06, 2.08592337e-06, 2.04526441e-06
            , 2.01427224e-06, 1.99367832e-06, 1.98408088e-06, 1.98593239e-06
            , 1.99953297e-06, 2.02501496e-06, 2.06234381e-06, 2.11130873e-06
            , 2.17152296e-06, -1.02076729e-05, -1.01268261e-05, -1.00369380e-05
            , -9.93907815e-06, -9.83446625e-06, -9.72446287e-06, -9.61055775e-06
            , -9.49434798e-06, -9.37752520e-06, -9.26185567e-06, -9.14915836e-06
            , -9.04128592e-06, -8.94010009e-06, -8.84744532e-06, -8.76513786e-06
            , -8.69492396e-06, -8.63847526e-06, -8.59735337e-06, -8.57299528e-06
            , -8.56668794e-06, -8.57955638e-06, -8.61253829e-06, -8.66637401e-06
            , -8.74159559e-06, -8.83850680e-06, -8.95718495e-06, -9.09746905e-06
            , -9.25895893e-06, -9.44101612e-06, -9.64276478e-06, -9.86309715e-06
            , -1.01006826e-05, 2.09611540e-06, 1.82884605e-06, 1.54952761e-06
            , 1.26024520e-06, 9.63229240e-07, 6.60824810e-07, 3.55485582e-07
            , 4.97236563e-08, -2.53896900e-07, -5.52788606e-07, -8.44385966e-07
            , -1.12615101e-06, -1.39560916e-06, -1.65038568e-06, -1.88820957e-06
            , -2.10695612e-06, -2.30467185e-06, -2.47958087e-06, -2.63011384e-06
            , -2.75492175e-06, -2.85289570e-06, -2.92317191e-06, -2.96514440e-06
            , -2.97847805e-06, -2.96309031e-06, -2.91917377e-06, -2.84719977e-06
            , -2.74787112e-06, -2.62215781e-06, -2.47127400e-06, -2.29664192e-06
            , -2.09990958e-06, -1.88289846e-06, -1.64762412e-06, -1.39622239e-06
            , -1.13096576e-06, -8.54226926e-07, -5.68443397e-07, -2.76101503e-07
            , 2.02931556e-08, 3.18244417e-07, 6.15292663e-07, 9.09034895e-07
            , 1.19715764e-06, 1.47745050e-06, 1.74783804e-06, 2.00639511e-06
            , -1.01987307e-05, -9.96892322e-06, -9.75565126e-06, -9.56008262e-06
            , -9.38316407e-06, -9.22562231e-06, -9.08795755e-06, -8.97044447e-06
            , -8.87313217e-06, -8.79585605e-06, -8.73823592e-06, -8.69969517e-06
            , -8.67946619e-06, -8.67660856e-06, -8.69002361e-06, -8.71847442e-06
            , -8.76060130e-06, -8.81494270e-06, -8.87996066e-06, -8.95405719e-06
            , -9.03559612e-06, -9.12293035e-06, -9.21441551e-06, -9.30843817e-06
            , -9.40342852e-06, -9.49788682e-06, -9.59039335e-06, -9.67963206e-06
            , -9.76439696e-06, -9.84361213e-06, -9.91633806e-06, -9.98177711e-06
            , -1.00392881e-05, -1.00883835e-05, -1.01287305e-05, -1.01601581e-05
            , -1.01826445e-05, -1.01963215e-05, -1.02014610e-05, -1.01984706e-05
            , -1.01878832e-05, -1.01703417e-05, -1.01465903e-05, -1.01174574e-05
            , -1.00838424e-05, -1.00466987e-05, -1.00070183e-05, -9.96581548e-06
            , -9.92410878e-06, -9.88290685e-06, -9.84319286e-06, -9.80590630e-06
            , -9.77193577e-06, -9.74209706e-06, -9.71712961e-06, -9.69768098e-06
            , -9.68429867e-06, -9.67742744e-06, -9.67739743e-06, -9.68442328e-06
            , -9.69860594e-06, -9.71992449e-06, -9.74824434e-06, -9.78331900e-06
            , -9.82479287e-06, -9.87220938e-06, -9.92501828e-06, -9.98258656e-06
            , -1.00442039e-05, -1.01091018e-05, -1.01764563e-05, -1.02454087e-05
            , 2.13502085e-06, 2.06554068e-06, 1.99714123e-06, 1.93070400e-06
            , 1.86708348e-06, 1.80709424e-06, 1.75150171e-06, 1.70100930e-06
            , 1.65624863e-06, 1.61777496e-06, 1.58605314e-06, 1.56146075e-06
            , 1.54427516e-06, 1.53467386e-06, 1.53273709e-06, 1.53844030e-06
            , 1.55166231e-06, 1.57218176e-06, 1.59968840e-06, 1.63378365e-06
            , 1.67398798e-06, 1.71974989e-06, 1.77045456e-06, 1.82543192e-06
            , 1.88396996e-06, 1.94531981e-06, 2.00871295e-06, 2.07336825e-06
            , 2.13850535e-06, -1.02467402e-05, -1.01829291e-05, -1.01208707e-05
            , -1.00612415e-05, -1.00046664e-05, -9.95171285e-06, -9.90288299e-06
            , -9.85860606e-06, -9.81923677e-06, -9.78505250e-06, -9.75624516e-06
            , -9.73292663e-06, -9.71512145e-06, -9.70277870e-06, -9.69576467e-06
            , -9.69386929e-06, -9.69681605e-06, -9.70425936e-06, -9.71579630e-06
            , -9.73097576e-06, -9.74930026e-06, -9.77023956e-06, -9.79323613e-06
            , -9.81771882e-06, -9.84310736e-06, -9.86882242e-06, -9.89429918e-06
            , -9.91898924e-06, -9.94237234e-06, -9.96396648e-06, -9.98333053e-06
            , -1.00000716e-05, -1.00138523e-05, -1.00243924e-05, -1.00314774e-05
            , -1.00349525e-05, -1.00347343e-05, -1.00308007e-05, -1.00231991e-05
            , -1.00120387e-05, -9.99749136e-06, -9.97978532e-06, -9.95920345e-06
            , -9.93607409e-06, -9.91077013e-06, -9.88369629e-06, -9.85528914e-06
            , -9.82600250e-06, -9.79630568e-06, -9.76667252e-06, -9.73757233e-06
            , -9.70946712e-06, -9.68279892e-06, -9.65798608e-06, -9.63541424e-06
            , -9.61543265e-06, -9.59834688e-06, -9.58441433e-06, -9.57384327e-06
            , -9.56678377e-06, -9.56333133e-06, -9.56352324e-06, -9.56733675e-06
            , -9.57469547e-06, -9.58546116e-06, -9.59944646e-06,
         -9.61641217e-06])
    dout_point = 0
    dout_channel = 0
    conv_index = 0
    plot_signal_time(np_dout,
                     title=f"dout_point {dout_point}, "
                     f"dout_channel {dout_channel},"
                     f" conv {conv_index}",
                     xlabel="Time")


def next_power2(x):
    """
    :param x: an integer number
    :return: the power of 2 which is the larger than x but the
    smallest possible

    >>> result = next_power2(5)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(1)
    >>> np.testing.assert_equal(result, 1)
    >>> result = next_power2(2)
    >>> np.testing.assert_equal(result, 2)
    >>> result = next_power2(7)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(9)
    >>> np.testing.assert_equal(result, 16)
    >>> result = next_power2(16)
    >>> np.testing.assert_equal(result, 16)
    >>> result = next_power2(64)
    >>> np.testing.assert_equal(result, 64)
    >>> result = next_power2(63)
    >>> np.testing.assert_equal(result, 64)
    """
    # return math.pow(2, math.ceil(math.log2(x)))
    return int(2 ** np.ceil(np.log2(x)))