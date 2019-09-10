import torch

from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import AttackType
from cnns.nnlib.utils.general_utils import AdversarialType
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import SchedulerType
from cnns.nnlib.utils.general_utils import LossType
from cnns.nnlib.utils.general_utils import LossReduction
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import Bool
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import PrecisionType
from cnns.nnlib.utils.general_utils import PredictionType
from cnns.nnlib.utils.general_utils import PolicyType

from cnns.nnlib.utils.general_utils import get_log_time

"""
cFFT cuda_multiply: total elapsed time (sec):  15.602577447891235

Pytorch: total elapsed time (sec):  7.639773607254028
"""

# 1D
# conv_type = ConvType.FFT1D
# conv_type = ConvType.STANDARD

# 2D
conv_type = ConvType.STANDARD2D
# conv_type = ConvType.FFT2D
compress_rate = 0.0

if conv_type == ConvType.FFT1D or conv_type == ConvType.STANDARD:
    # dataset = "ucr"
    # dataset = "WIFI64"
    # dataset = "debug22"  # only Adiac
    # dataset = "WIFI5-192"
    # dataset = "WIFI"
    # dataset = "WIFI_class_3_sample_512"
    # dataset = '2_classes_WiFi'
    # dataset = 'CaseC_los'
    dataset = 'deeprl'
    # network_type = NetworkType.FCNN_STANDARD
    # network_type = NetworkType.Linear3
    # network_type = NetworkType.VGG1D_7
    # network_type = NetworkType.VGG1D_6
    # network_type = NetworkType.FCNN_VERY_TINY
    # network_type = NetworkType.FCNN_MICRO
    network_type = NetworkType.Linear4
    preserved_energy = 100  # for unit tests
    # learning_rate = 0.0005
    learning_rate = 0.001
    batch_size = 32
    test_batch_size = batch_size
    # test_batch_size = 256
    weight_decay = 0.0001
    # weight_decay = 0.0
    # weight_decay = 0.01
    preserved_energies = [preserved_energy]
    tensor_type = TensorType.FLOAT32
    precision_type = PrecisionType.FP32
    # conv_exec_type = ConvExecType.BATCH
    conv_exec_type = ConvExecType.CUDA
    visualize = False  # test model for different compress rates
    next_power2 = True
    schedule_patience = 50
    schedule_factor = 0.5
    epochs = 10
    optimizer_type = OptimizerType.ADAM
    momentum = 0.9
    # loss_type = LossType.CROSS_ENTROPY
    loss_type = LossType.MSE
    loss_reduction = LossReduction.MEAN
    model_path = "no_model"
    # model_path = 'wifi-all-accuracy-99-25.model'
    # model_path = 'pytorch_behave1.model'
    in_channels = 1
else:
    # dataset = "mnist"
    # dataset = "cifar10"
    # dataset = "cifar100"
    dataset = "imagenet"
    # dataset = "svhn"

    batch_size = 32
    # test_batch_size = batch_size
    # test_batch_size = 256
    test_batch_size = batch_size
    learning_rate = 0.01
    if dataset == 'imagenet':
        learning_rate = 0.1
    weight_decay = 0.0001
    momentum = 0.9
    # epochs = 14101
    # epochs=50
    # epochs = 101
    # epochs = 100
    # epochs = 50000
    # epochs = 250
    epochs = 10001
    preserved_energy = 100  # for unit tests
    preserved_energies = [preserved_energy]
    tensor_type = TensorType.FLOAT32
    precision_type = PrecisionType.FP32
    conv_exec_type = ConvExecType.CUDA
    # conv_exec_type = ConvExecType.SGEMM
    visualize = False  # test model for different compress rates
    next_power2 = True
    schedule_patience = 10
    schedule_factor = 0.5
    optimizer_type = OptimizerType.MOMENTUM
    loss_type = LossType.CROSS_ENTROPY
    loss_reduction = LossReduction.MEAN
    in_channels = 3

    if dataset == "mnist":
        batch_size = 64
        test_batch_size = 1000
        momentum = 0.5
        learning_rate = 0.01
        epochs = 100
        loss_type = LossType.NLL
        loss_reduction = LossReduction.SUM
        network_type = NetworkType.Net
        model_path = "2019-05-03-10-08-51-149612-dataset-mnist-preserve-energy-100-compress-rate-0.0-test-accuracy-99.07-channel-vals-0.model"
    elif dataset == "cifar10":
        network_type = NetworkType.ResNet18
        # model_path = "saved_model_2019-04-08-16-51-16-845688-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.22-channel-vals-0.model"
        model_path = "saved_model_2019-05-16-11-37-45-415722-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.56-channel-vals-0.model"
        # model_path = "no_model"
        # model_path = 'saved-model-2019-08-13-23-06-55-540595-dataset-cifar10-preserve-energy-100-compress-rate-0.0-test-loss-0.01350257922783494-test-accuracy-91.83-channel-vals-0-fft-channel-50-percent.model'
        # model_path = 'saved_model_2019-08-13-21-50-15-942405-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-loss-0.023045711521059276-test-accuracy-86.73-channel-vals-0-fft-layer-85-percent.model'
    elif dataset == "cifar100":
        network_type = NetworkType.DenseNetCifar
        weight_decay = 0.0001
        model_path = "no_model"
    elif dataset == "imagenet":
        network_type = NetworkType.ResNet50
        batch_size = 1
        test_batch_size = batch_size
        learning_rate = 0.1
        weight_decay = 0.0001
        model_path = "pretrained"
    elif dataset == "svhn":
        network_type = NetworkType.ResNet18
        model_path = "no_model"
    elif dataset.startswith("WIFI"):
        network_type = NetworkType.FCNN_STANDARD
        model_path = "no_model"
    else:
        raise Exception(f"Unknown dataset name: {dataset}")

import os

try:
    USER = os.environ['USER']
except KeyError:
    USER = "adzie"


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
                 is_debug=True,
                 # network_type=NetworkType.ResNet18,
                 # network_type=NetworkType.DenseNetCifar,
                 network_type=network_type,
                 preserved_energy=preserved_energy,  # for unit tests
                 preserved_energies=preserved_energies,
                 # preserved_energies=range(100,49,-1),
                 # preserved_energies=range(85, 75, -1),
                 # preserved_energies=[95, 90, 98],                 # preserved_energies=[100, 99.9, 99.5, 99, 98, 97, 96, 95, 90, 80, 70, 60, 50, 10],
                 # preserved_energies=[100,99.5,99,98,95,90,80],
                 # preserved_energies=range(96, 1),
                 # tensor_type=TensorType.FLOAT16,
                 tensor_type=tensor_type,
                 # precision_type=PrecisionType.AMP,  # use AMP for fp16 - reduced precision training
                 precision_type=PrecisionType.FP32,
                 # precision_type=PrecisionType.FP16,
                 use_cuda=True,
                 compress_type=CompressType.STANDARD,
                 compress_rate=compress_rate,
                 compress_rates=[compress_rate],
                 # compress_rates=[75, 50, 10, 1],
                 # compress_rates=[0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1],
                 # ndexes_back=[5,15,25,35,45],
                 # compress_rates=range(80, 101, 10),
                 # compress_rates=[x/2 for x in range(28,111,1)],
                 # compress_rate=0.1,  # for unit tests
                 # compress_rates=[84.0], # 47.5, 84.0
                 # compress_rates=range(85, 91),
                 # compress_rates=[50.0],
                 # compress_rates=[90.0],
                 # compress_rates=[0,1,6,10,12,20,22,28,38,47,48,76,84],
                 # compress_rates=[0,3,4,9.5,11.5,12,17.5,22,22.5,28,37.5,47,47.5,51,64.5,74,77.5,84],
                 # compress_rates=range(0,30),
                 # compress_rates=[0,5,11,11.5,17,20.5,22,22.5,28,32,33,36,37,39,41,42,47,50,51,55,58,59,63,64,65,66,69,70,71,73,76,77,79,80,82,83,84],
                 # compress_rates = [0,0,3,4,9.5,11.5,11.5,12,17.5,22,22.5,28,37.5,47,47.5,51,64.5,74,77.5,84],
                 # layers_compress_rates=None,
                 # compression rates for each of the conv fft layers in
                 # ResNet-18, with the total compression in the fft domain by
                 # more than 50%, about 92% of the energy is preserved
                 # layers_compress_rates=[72.62019231,70.17045455,73.4375,76.51515152,76.51515152,79.40340909,30.51470588,30.51470588,40.25735294,40.25735294,0,0,0,16.66666667,0,0,0],
                 layers_compress_rates=None,
                 # weight_decay=5e-4,
                 # weight_decay=0,
                 weight_decay=weight_decay,
                 epochs=epochs,
                 min_batch_size=batch_size,
                 test_batch_size=test_batch_size,
                 learning_rate=learning_rate,
                 momentum=momentum,
                 seed=31,
                 log_interval=1,
                 optimizer_type=optimizer_type,
                 scheduler_type=SchedulerType.ReduceLROnPlateau,
                 # scheduler_type=SchedulerType.Custom,
                 loss_type=loss_type,
                 loss_reduction=loss_reduction,
                 memory_type=MemoryType.PINNED,
                 workers=4,
                 model_path=model_path,
                 # model_path="no_model",
                 # model_path="2019-05-03-10-08-51-149612-dataset-mnist-preserve-energy-100-compress-rate-0.0-test-accuracy-99.07-channel-vals-0.model",
                 # ="saved_model_2019-04-08-16-51-16-845688-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.22-channel-vals-0.model",
                 # model_path = "2019-04-29-08-31-35-212961-dataset-cifar10-preserve-energy-100-compress-rate-0.1-test-accuracy-84.27-channel-vals-8.model",
                 # model_path="2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model",
                 # model_path="2019-01-08-14-41-44-026589-dataset-cifar10-preserve-energy-100.0-test-accuracy-91.39-fp16-amp-no-compression.model",
                 # model_path="2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model",
                 # model_path="2019-01-12-23-31-40-502439-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.63-compress-20-percent-combine_energy-index_back.model",
                 # model_path="2019-01-11-02-21-05-406721-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.23-51.5-real-compression.model",
                 # model_path="2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model",
                 # model_path="2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model",
                 # model_path="2019-01-08-02-48-26-558883-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.07-compress-8.32.model",
                 # model_path="2019-01-08-07-18-52-980249-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.82-compress-29.26.model",
                 # model_path="2019-01-18-20-11-05-764539-dataset-cifar100-preserve-energy-100.0-test-accuracy-58.63.model",
                 # model_path="2019-01-09-14-51-48-093159-dataset-ECGFiveDays-preserve-energy-100.0-test-accuracy-86.52729384436701.model",
                 # model_path="2019-01-09-13-46-20-792223-dataset-MoteStrain-preserve-energy-90.0-test-accuracy-79.47284345047923.model",
                 # model_path="2019-01-09-13-47-45-327309-dataset-MoteStrain-preserve-energy-100.0-test-accuracy-80.59105431309904.model",
                 # model_path="2019-01-09-07-13-27-296552-dataset-Cricket_Z-preserve-energy-90.0-test-accuracy-62.56410256410256.model",
                 # model_path="2019-01-09-08-07-40-552436-dataset-Cricket_Z-preserve-energy-100.0-test-accuracy-61.282051282051285.model",
                 # model_path="2019-01-09-14-05-53-114845-dataset-Cricket_X-preserve-energy-90.0-test-accuracy-59.743589743589745.model",
                 # model_path="2019-01-09-05-47-48-678983-dataset-Cricket_X-preserve-energy-100.0-test-accuracy-60.256410256410255.model",
                 # model_path="2019-01-09-15-01-33-103045-dataset-Cricket_Y-preserve-energy-90.0-test-accuracy-56.666666666666664.model",
                 # model_path="2019-01-09-14-56-29-165361-dataset-Cricket_Y-preserve-energy-100.0-test-accuracy-55.38461538461539.model",
                 # model_path="2019-01-09-04-51-21-547489-dataset-Strawberry-preserve-energy-90.0-test-accuracy-80.75040783034258.model",
                 # model_path="2019-01-09-05-04-18-678916-dataset-Strawberry-preserve-energy-100.0-test-accuracy-84.66557911908646.model",
                 # model_path="2019-01-09-09-12-35-908217-dataset-FacesUCR-preserve-energy-90.0-test-accuracy-86.0.model",
                 # model_path="2019-01-08-23-47-34-143794-dataset-FacesUCR-preserve-energy-100.0-test-accuracy-84.4390243902439.model",
                 # model_path="2019-01-09-07-46-23-106946-dataset-FaceFour-preserve-energy-90.0-test-accuracy-78.4090909090909.model",
                 # model_path="2019-01-09-07-58-50-217192-dataset-FaceFour-preserve-energy-100.0-test-accuracy-82.95454545454545.model",
                 # model_path="2019-01-09-14-42-37-071424-dataset-ECGFiveDays-preserve-energy-90.0-test-accuracy-84.3205574912892.model",
                 # model_path="2019-01-09-16-57-28-772608-dataset-MiddlePhalanxOutlineAgeGroup-preserve-energy-100.0-test-accuracy-78.75.model",
                 # model_path="2019-01-09-04-53-39-014229-dataset-MiddlePhalanxOutlineAgeGroup-preserve-energy-90.0-test-accuracy-77.25.model",
                 # model_path="2019-01-08-23-06-38-689920-dataset-ToeSegmentation1-preserve-energy-100.0-test-accuracy-91.2280701754386.model",
                 # model_path="2018-12-11-09-55-48-256434-dataset-Plane-preserve-energy-100.0-test-accuracy-99.04761904761905.model",
                 # model_path="2019-01-08-23-05-54-122929-dataset-ToeSegmentation1-preserve-energy-90.0-test-accuracy-91.66666666666667.model",
                 # model_path="2019-01-08-23-51-49-011087-dataset-Trace-preserve-energy-100.0-test-accuracy-100.0.model",
                 # model_path="2019-01-08-23-51-27-260351-dataset-Trace-preserve-energy-90.0-test-accuracy-100.0.model",
                 # model_path="2018-11-25-00-36-07-133900-dataset-ItalyPowerDemand-preserve-energy-100-test-accuracy-95.91836734693878.model",
                 # model_path="2018-11-25-11-10-07-508525-dataset-Lighting7-preserve-energy-100-test-accuracy-82.1917808219178.model",
                 # model_path="2018-12-01-03-23-06-181637-dataset-Two_Patterns-preserve-energy-99-test-accuracy-88.25.model",
                 # model_path="2018-11-30-21-58-26-723085-dataset-Two_Patterns-preserve-energy-90-test-accuracy-93.8.model",
                 # model_path="2018-12-01-03-00-20-144358-dataset-Two_Patterns-preserve-energy-100-test-accuracy-87.15.model",
                 # model_path="2018-12-02-19-31-12-260418-dataset-yoga-preserve-energy-100-test-accuracy-76.33333333333333.model",
                 # model_path="2018-12-01-14-52-20-121380-dataset-uWaveGestureLibrary_Z-preserve-energy-90-test-accuracy-70.54718034617532.model",
                 # model_path="2018-12-02-03-48-52-574269-dataset-uWaveGestureLibrary_Z-preserve-energy-99-test-accuracy-70.18425460636516.model",
                 # model_path="2018-12-05-10-16-49-256641-dataset-uWaveGestureLibrary_Z-preserve-energy-100-test-accuracy-70.2680067001675.model",
                 # model_path="2018-12-01-19-49-37-842889-dataset-yoga-preserve-energy-90-test-accuracy-69.53333333333333.model",
                 # model_path="2018-12-02-09-33-56-151932-dataset-yoga-preserve-energy-99-test-accuracy-71.53333333333333.model",
                 # model_path="2018-11-27-06-17-22-024663-dataset-yoga-preserve-energy-100-test-accuracy-75.33333333333333.model",
                 # model_path="2018-11-29-12-50-03-178526-dataset-50words-preserve-energy-100-test-accuracy-0.0.model",
                 # model_path="2018-11-29-00-08-38-530297-dataset-SwedishLeaf-preserve-energy-100-test-accuracy-94.56.model",
                 # model_path="2018-11-29-11-28-09-977656-dataset-50words-preserve-energy-90-test-accuracy-59.56043956043956.model",
                 # model_path="2018-11-29-13-12-20-114486-dataset-50words-preserve-energy-100-test-accuracy-63.51648351648352.model",
                 # model_path="2018-11-29-12-26-08-403300-dataset-50words-preserve-energy-99-test-accuracy-63.51648351648352.model",
                 # model_path="2018-11-26-20-04-34-197804-dataset-50words-preserve-energy-100-test-accuracy-67.47252747252747.model",
                 # model_path="2019-01-11-02-21-05-406721-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.23-51.5-real-compression.model",
                 # dataset="cifar100",
                 # dataset="cifar10",
                 dataset=dataset,
                 # dataset="debug",
                 mem_test=False,
                 is_data_augmentation=True,
                 sample_count_limit=0,  # 0 means run on full data
                 # sample_count_limit=1024,
                 # sample_count_limit = 100,
                 # sample_count_limit=32,
                 # sample_count_limit=100,
                 # sample_count_limit=1024,
                 # sample_count_limit=2048,
                 # conv_type=ConvType.FFT2D,
                 # conv_type=ConvType.STANDARD2D,
                 conv_type=conv_type,
                 # conv_type=ConvType.STANDARD,
                 # conv_exec_type=ConvExecType.CUDA,
                 # conv_exec_type=ConvExecType.CUDA_DEEP,
                 # conv_exec_type=ConvExecType.CUDA_SHARED_LOG,
                 conv_exec_type=conv_exec_type,
                 # conv_exec_type=ConvExecType.SERIAL,
                 visualize=visualize,  # test model for different compress rates
                 static_loss_scale=1,
                 out_size=None,
                 next_power2=next_power2,
                 dynamic_loss_scale=True,
                 memory_size=25,
                 is_progress_bar=False,
                 log_conv_size=False,
                 stride_type=StrideType.STANDARD,
                 # is_dev_dataset = True,
                 is_dev_dataset=False,
                 dev_percent=0,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 cuda_block_threads=1024,
                 # resume="cifar100-90.484checkpoint.tar",
                 # resume="cifar100-0.0-84-checkpoint.tar",
                 resume="",
                 gpu=0,
                 # start_epoch=14100,
                 # start_epoch=50,
                 start_epoch=10000,
                 # start_epoch=249,
                 only_train=False,
                 test_compress_rates=False,
                 noise_sigma=0.0,
                 # noise_sigmas=[0.0],
                 # noise_sigmas=[x/1000 for x in range(1, 10)] + [x/100 for x in range(1, 10)] + [x/10 for x in range(1, 11)],
                 noise_sigmas=[0.03],
                 # noise_sigmas=[0.05, 0.06, 0.07, 0.08, 0.09],
                 # noise_sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 # noise_epsilons=[0.1, 0.07, 0.03, 0.009, 0.007, 0.04, 0.02, 0.3],
                 fft_type="real_fft",  # real_fft or complex_fft
                 imagenet_path="/home/" + str(USER) + "/imagenet",
                 distributed=False,
                 in_channels=in_channels,
                 values_per_channel=0,
                 many_values_per_channel=[0],
                 # ucr_path="../sathya/ML_NLOS/15F_NLOS/",
                 ucr_path="../sathya/CaseC/",
                 # ucr_path="../../TimeSeriesDatasets",
                 start_epsilon=0,
                 # attack_type=AttackType.BAND_ONLY,
                 # attack_type=AttackType.NOISE_ONLY,
                 # attack_type=AttackType.ROUND_ONLY,
                 # attack_type=AttackType.FFT_RECOVERY,
                 attack_type=AttackType.RECOVERY,
                 # attack_type=AttackType.ROUND_RECOVERY,
                 # attack_type=AttackType.LAPLACE_ONLY,
                 # attack_type=AttackType.GAUSS_ONLY,
                 schedule_patience=schedule_patience,
                 schedule_factor=schedule_factor,
                 compress_fft_layer=0,
                 # attack_name="CarliniWagnerL2AttackRoundFFT",
                 attack_name="CarliniWagnerL2Attack",
                 # attack_name=None,
                 # attack_name="FGSM",
                 # attack_name="GaussAttack",
                 # attack_name="FFTHighFrequencyAttack",
                 # attack_name="FFTHighFrequencyAttackAdversary",
                 # attack_name="FFTLimitFrequencyAttack",
                 # attack_name="FFTLimitFrequencyAttackAdversary",
                 # attack_name="FFTReplaceFrequencyAttack",
                 # attack_name="FFTSingleFrequencyAttack",
                 # attack_name="FFTMultipleFrequencyAttack",
                 # attack_name="FFTSmallestFrequencyAttack",
                 # attack_name="FFTLimitValuesAttack",
                 # attack_name="FFTLimitMagnitudesAttack",
                 # attack_name="FFTMultipleFrequencyBinarySearchAttack",
                 # attack_name="EmptyAttack",
                 # attack_name="Nattack",
                 # attack_name="SimbaSingle",
                 interpolate="const",
                 # recover_type="rounding",
                 # recover_type="fft",
                 # recover_type="all",
                 # recover_type="laplace",
                 # recover_type="debug",
                 recover_type="gauss",
                 # recover_type="empty",
                 noise_epsilon=0.0,
                 noise_epsilons=[0.0],
                 step_size=50,
                 noise_iterations=0,
                 many_noise_iterations=[0],
                 recover_iterations=0,
                 many_recover_iterations=[0],
                 attack_max_iterations=0,
                 many_attack_iterations=[1000],
                 laplace_epsilon=0.0,
                 laplace_epsilons=[0.0],
                 is_DC_shift=False,
                 use_foolbox_data=False,
                 svd_compress=0.0,
                 many_svd_compress=[0.0],
                 adv_type=AdversarialType.BEFORE,
                 # adv_type=AdversarialType.NONE,
                 # prediction_type=PredictionType.REGRESSION,
                 prediction_type=PredictionType.CLASSIFICATION,
                 # 'regression' or 'classification'
                 # attack_strengths=[0.01, 0.03, 0.04, 0.05, 0.07, 0.1, 0.5, 1.0],
                 # attack_strengths=[0.08, 0.09, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                 attack_strengths=(0.01, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0,),
                 # attack_strengths=(0.0,),
                 ):
        """
        The default parameters for the execution of the program.

        :param is_debug: should we print the debug messages.
        :param network_type: the type of network architecture
        :param preserve_energy: how much energy in the input should be preserved
        after compression
        :param use_cuda: should use gpu or not
        :param compress_type: the type of FFT compression, NO_FILTER - do not
        compress the filter. BIG_COEF: preserve only the largest coefficients
        in the frequency domain.
        :param compress_rate: how much compress based on discarded coefficients
        in the frequency domain.
        :param layers_compress_rates: the compression rate for each of the fft
        based convolution layers.
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
        :param log_conv_size: log the size of the convolutional layers
        :param is_dev_set: is the dev dataset used (extracted from the trina set)
        :param dev_percent: % of data used from the train set as the dev set
        :param is_serial_conv: is the convolution exeucted as going serially
        through all data points, and convolving separately each data point with
        all filters all a few datapoints with all filters in one go
        """
        super(Arguments).__init__()
        self.__counter__ = 0

        self.is_debug = is_debug
        self.__idx_is_debug = self.__next_counter()

        self.network_type = network_type
        self.__idx_network_type = self.__next_counter()

        if use_cuda is None:
            self.use_cuda = True if torch.cuda.is_available() else False
        else:
            self.use_cuda = use_cuda
        self.__idx_use_cuda = self.__next_counter()

        self.compress_type = compress_type
        self.__idx_compress_type = self.__next_counter()

        self.weight_decay = weight_decay
        self.__idx_weight_decay = self.__next_counter()

        # Object's variable that are not convertible to an element of a tensor.
        self.preserve_energy = preserved_energy  # for unit tests
        self.preserve_energies = preserved_energies
        self.compress_rate = compress_rate  # for unit tests
        self.compress_rates = compress_rates
        self.layers_compress_rates = layers_compress_rates
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
        self.conv_exec_type = conv_exec_type
        self.visulize = visualize
        self.static_loss_scale = static_loss_scale
        self.out_size = out_size
        self.tensor_type = tensor_type
        self.next_power2 = next_power2
        self.dynamic_loss_scale = dynamic_loss_scale
        self.memory_size = memory_size
        self.is_progress_bar = is_progress_bar
        self.log_conv_size = log_conv_size
        self.stride_type = stride_type
        self.is_dev_dataset = is_dev_dataset
        self.dev_percent = dev_percent
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.cuda_block_threads = cuda_block_threads
        self.resume = resume
        self.gpu = gpu
        self.start_epoch = start_epoch
        self.precision_type = precision_type
        self.only_train = only_train
        self.test_compress_rates = test_compress_rates
        self.noise_sigma = noise_sigma
        self.noise_sigmas = noise_sigmas
        self.fft_type = fft_type
        self.imagenet_path = imagenet_path
        self.distributed = distributed
        self.in_channels = in_channels
        self.values_per_channel = values_per_channel
        self.many_values_per_channel = many_values_per_channel
        self.ucr_path = ucr_path
        self.start_epsilon = start_epsilon
        self.attack_type = attack_type
        self.schedule_factor = schedule_factor
        self.schedule_patience = schedule_patience
        self.compress_fft_layer = compress_fft_layer
        self.attack_name = attack_name
        self.noise_epsilon = noise_epsilon
        self.noise_epsilons = noise_epsilons
        self.interpolate = interpolate
        self.recover_type = recover_type
        self.step_size = step_size
        self.noise_iterations = noise_iterations
        self.many_noise_iterations = many_noise_iterations
        self.recover_iterations = recover_iterations
        self.many_recover_iterations = many_recover_iterations
        self.attack_max_iterations = attack_max_iterations
        self.many_attack_iterations = many_attack_iterations
        self.laplace_epsilon = laplace_epsilon
        self.laplace_epsilons = laplace_epsilons
        self.is_DC_shift = is_DC_shift
        self.use_foolbox_data = use_foolbox_data
        self.svd_compress = svd_compress
        self.many_svd_compress = many_svd_compress
        self.adv_type = adv_type
        self.prediction_type = prediction_type
        self.attack_strengths = attack_strengths
        self.targeted_attack = True

        # deeprl
        # self.env_name = "Reacher-v2"
        self.env_name = "Ant-v2"
        # self.env_name = "Hopper-v2"
        self.expert_data_dir = 'expert_data/'
        self.dagger_data_dir = 'dagger_data/'
        self.behave_model_prefix = 'behave_models/'
        self.dagger_model_prefix = 'dagger_models/'
        self.hidden_units = 64
        # train_steps = 1000000
        self.train_steps = 0
        self.rollouts = 10
        self.verbose = False
        self.max_timesteps = None
        self.render = False
        self.policy_type = PolicyType.PYTORCH_BEHAVE
        # self.policy_type = PolicyType.PYTORCH_DAGGER
        # self.policy_type = PolicyType.EXPERT
        # self.policy_type = PolicyType.TENSORFLOW_BEHAVE
        # self.learn_policy_file = self.get_model_file()
        # self.learn_policy_file = 'dagger_models/2019-08-06-13-13-11-570969_env_name_Ant-v2_return_825.7569984208542_train_loss_3.320436139918985e-05_test_loss_3.113336426401105e-05_.model'
        # self.learn_policy_file = 'dagger_models/2019-08-06-13-46-25-789767_env_name_Ant-v2_return_572.6525353133684_train_loss_4.008859843047313e-05_test_loss_3.9420387411200985e-05_.model'
        # self.learn_policy_file = 'dagger_models/2019-08-06-16-20-07-791337_env_name_Ant-v2_return_338.0155704855727_train_loss_5.063490920386164e-05_test_loss_4.702933799257026e-05_.model'
        # self.learn_policy_file = 'dagger_models/2019-08-06-19-22-45-869247_env_name_Ant-v2_return_1629.5950472163659_train_loss_6.733186626618934e-05_test_loss_4.8252221797972436e-05_.model'
        # self.learn_policy_file = 'behave_models/Ant-v2-2000-epoch-94-return819.72-std-11.51.model'
        # self.learn_policy_file = 'behave_models/Ant-v2-1500-epoch-94-return-805.67-std-19.45.model'
        self.learn_policy_file = 'behave_models/Ant-v2-2000-epoch-27-return-809.76-std-15.29.model'
        # self.learn_policy_file = 'behave_models/2019-08-06-12-24-43-113461_env_name_Ant-v2_rollouts_2000_epoch_4_train_loss_3.362916449174726e-05_test_loss_3.3841744545288605e-05.model'
        # self.learn_policy_file = 'behave_models/saved-model-reacher-v2-10000-rolls-loss-1.99.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-3.05.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-1.96.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-2.6-100rolls.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-0.00015-100.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-0.00024-10rolls.model'
        # self.learn_policy_file = 'behave_models/reacher-v2-test-loss-0.0014-1-roll.model'
        # self.learn_policy_file = 'behave_models/hopper-test-loss-5.023237937094757e-05.model'
        # self.learn_policy_file = 'behave_models/hopper-test-loss-5.02-05.model'
        # self.learn_policy_file = 'behave_models/hopper-test-loss-2.45-05.model'
        # self.learn_policy_file = 'dagger_models/2019-07-24-16-15-26-243827_return_61.58420190905099_train_loss_0.023740299610477505_test_loss_0.15592382848262787_.model'
        # self.learn_poicy_file = 'no_policy_file'
        # self.learn_policy_file = 'models/pytorch_behave.model'
        # self.learn_policy_file = 'dagger_models/2019-07-22-15-02-02-783627_return_-8.853555681674216_train_loss_1.1837590678164633e-05_test_loss_6.563135706418314e-06_.model'
        # self.learn_policy_file = 'dagger_modes/2019-07-22-16-52-59-627577_return_-8.284563905838292_train_loss_1.4905159095937342e-05_test_loss_6.925928364106574e-06_.model'
        # self.learn_policy_file = 'dagger_models/2019-07-23-12-03-47-000774_return_-7.953395893599401_train_loss_5.9503303005835296e-05_test_loss_8.228212310429874e-06_.model'
        self.expert_policy_file = "experts/" + self.env_name + ".pkl"
        # self.rollout_file = '../nnlib/datasets/deeprl/data/' + self.env_name + '-1000.pkl'
        # self.rollout_file = '../../deeprl/expert_data/' + self.env_name + '-1000.pkl'
        # self.rollout_file = 'expert_data/' + self.env_name + '-100.pkl'
        # self.rollout_file = 'expert_data/Ant-v2_rollouts_2000_mean-return_4770.21597610086_std-return_401.4036761529797_.pkl'
        # self.rollout_file = 'dagger_data/Ant-v2-0-2019-08-06-12-13-20.pkl'
        # self.rollout_file = 'dagger_data/Ant-v2-0-2019-08-06-15-10.pkl'
        # self.rollout_file = 'dagger_data/Ant-v2-20-2019-08-06-17-20.pkl'
        self.rollout_file = 'dagger_data/Ant-v2-70.pkl'
        # self.rollout_file = 'dagger_data/' + self.env_name + '-600.pkl'
        self.dagger_iterations = 100
        self.behave_iterations = 100

        self.log_file = 'logs/' + get_log_time() + '-log' + '.txt'
        self.delimiter = ";"
        self.pickle_protocol = 2

        self.set_dtype()
        self.set_device()

    def set_device(self):
        if torch.cuda.is_available() and self.use_cuda:
            # print("cuda is available: ")
            self.device = torch.device("cuda")
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")

    def get_model_file(self):
        if self.verbose:
            print('model file rollouts: ', self.rollouts)
            print('model file train steps: ', self.train_steps)

        if self.policy_type == PolicyType.TENSORFLOW_BEHAVE:
            model_file = self.behave_model_prefix + self.env_name + '-rollouts' + str(
                self.rollouts) + '-train-steps-' + str(
                self.train_steps) + '.ckpt'
        elif self.policy_type == PolicyType.EXPERT:
            model_file = "experts/" + self.env_name + ".pkl"
        elif self.policy_type == PolicyType.PYTORCH_BEHAVE:
            model_number = 1
            # model_file = '../nnlib/pytorch_experiments/models/pytorch_behave' + str(
            #     model_number) + '.model'
            model_file = 'no_policy_file'
        elif self.policy_type == PolicyType.PYTORCH_DAGGER:
            model_file = 'no_policy_file'
        else:
            raise Exception(f'Unknown model type: {self.policy_type.name}')
        return model_file

    def get_bool(self, arg):
        return True if Bool[arg] is Bool.TRUE else False

    def set_parsed_args(self, parsed_args):
        # Make sure you do not miss any properties.
        # https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
        # self.__dict__ = parsed_args.__dict__.copy()
        parsed_dict = parsed_args.__dict__.copy()
        for arg in parsed_dict.keys():
            self.__dict__[arg] = parsed_dict[arg]

        # Enums:
        self.network_type = NetworkType[parsed_args.network_type]
        self.compress_type = CompressType[parsed_args.compress_type]
        self.conv_type = ConvType[parsed_args.conv_type]
        self.conv_exec_type = ConvExecType[parsed_args.conv_exec_type]
        self.optimizer_type = OptimizerType[parsed_args.optimizer_type]
        self.scheduler_type = SchedulerType[parsed_args.scheduler_type]
        self.loss_type = LossType[parsed_args.loss_type]
        self.loss_reduction = LossReduction[parsed_args.loss_reduction]
        self.memory_type = MemoryType[parsed_args.memory_type]
        self.tensor_type = TensorType[parsed_args.tensor_type]
        self.stride_type = StrideType[parsed_args.stride_type]
        self.precision_type = PrecisionType[parsed_args.precision_type]
        self.attack_type = AttackType[parsed_args.attack_type]
        self.adv_type = AdversarialType[parsed_args.adv_type]
        self.prediction_type = PredictionType[parsed_args.prediction_type]
        self.policy_type = PolicyType[parsed_args.policy_type]

        # Bools:
        self.is_debug = self.get_bool(parsed_args.is_debug)
        self.dynamic_loss_scale = self.get_bool(parsed_args.dynamic_loss_scale)
        self.next_power2 = self.get_bool(parsed_args.next_power2)
        self.visulize = self.get_bool(parsed_args.visualize)
        self.is_progress_bar = self.get_bool(parsed_args.is_progress_bar)
        self.log_conv_size = self.get_bool(parsed_args.log_conv_size)
        self.is_data_augmentation = self.get_bool(
            parsed_args.is_data_augmentation)
        self.is_dev_dataset = self.get_bool(parsed_args.is_dev_dataset)
        self.mem_test = self.get_bool(parsed_args.mem_test)
        self.use_cuda = self.get_bool(
            parsed_args.use_cuda) and torch.cuda.is_available()
        self.only_train = self.get_bool(parsed_args.only_train)
        self.test_compress_rates = self.get_bool(
            parsed_args.test_compress_rates)
        self.distributed = self.get_bool(parsed_args.distributed)
        self.is_DC_shift = self.get_bool(parsed_args.is_DC_shift)
        self.use_foolbox_data = self.get_bool(parsed_args.use_foolbox_data)

        if hasattr(parsed_args, "preserve_energy"):
            self.preserve_energy = parsed_args.preserve_energy

        self.set_dtype()
        self.set_device()

    def set_dtype(self):
        tensor_type = self.tensor_type
        if tensor_type is TensorType.FLOAT32:
            dtype = torch.float32
        elif tensor_type is TensorType.FLOAT16 or precision_type is PrecisionType.FP16:
            dtype = torch.float16
        elif tensor_type is TensorType.DOUBLE:
            dtype = torch.double
        else:
            raise Exception(f"Unknown tensor type: {tensor_type}")
        self.dtype = dtype

    def get_str(self):
        args_dict = self.__dict__
        args_str = " ".join(
            ["--" + str(key) + "=" + str(value) for key, value in
             sorted(args_dict.items())])
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
        t[self.__idx_index_back] = self.from_float_arg(self.compress_rate)

    def from_tensor(self, t):
        self.network_type = NetworkType(t[int(self.__idx_nework_type)])
        self.is_debug = self.to_bool_arg(t[self.__idx_is_debug])
        self.use_cuda = self.to_bool_arg(t[self.__idx_use_cuda])
        self.compress_type = CompressType(t[int(self.__idx_compress_type)])
        self.preserve_energy = self.to_float_arg(t[self.__idx_preserve_energy])
        self.compress_rate = self.to_float_arg(t[self.__idx_index_back])


if __name__ == "__main__":
    args = Arguments()
    print(args.get_str())
