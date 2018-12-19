import logging
import unittest
import gc
import numpy as np
import torch
from torch import tensor
from scipy import stats
from torch import tensor

from cnns.nnlib.layers import conv_backward_naive_1D
from cnns.nnlib.layers import conv_forward_naive_1D
from cnns.nnlib.pytorch_layers.conv1D_fft \
    import Conv1dfftFunction, Conv1dfftAutograd, Conv1dfft, Conv1dfftSimple, \
    Conv1dfftCompressSignalOnly, Conv1dfftSimpleForLoop
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.pytorch_layers.pytorch_utils import get_tensors
from cnns.nnlib.pytorch_layers.pytorch_utils import del_object
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.utils.arguments import Arguments

from cnns.nnlib.utils.general_utils import CompressType

ERR_MSG = "Expected x is different from computed y."

# convTypes = [Conv1dfftAutograd, Conv1dfft, Conv1dfftSimple,
#              Conv1dfftSimpleForLoop, Conv1dfftCompressSignalOnly]
convTypes = [Conv1dfft]

fifty_words = np.array([[[-0.89094, -0.86099, -0.82438, -0.78214, -0.73573,
                          -0.68691, -0.63754, -0.58937, -0.54342, -0.50044,
                          -0.46082, -0.42469, -0.3924, -0.36389, -0.33906,
                          -0.31795, -0.30056, -0.28692, -0.27727, -0.27105,
                          -0.26777, -0.26678, -0.26742, -0.26927, -0.27202,
                          -0.27529, -0.27898, -0.2832, -0.28814, -0.29431,
                          -0.30185, -0.31086, -0.32135, -0.33326, -0.3465,
                          -0.36103, -0.37659, -0.39297, -0.40995, -0.42731,
                          -0.44484, -0.46226, -0.47921, -0.49534, -0.51031,
                          -0.52382, -0.53567, -0.54568, -0.55375, -0.55984,
                          -0.56397, -0.56617, -0.56653, -0.56511, -0.56201,
                          -0.55736, -0.55133, -0.54419, -0.53617, -0.52758,
                          -0.51876, -0.51004, -0.50174, -0.49418, -0.48731,
                          -0.48128, -0.47619, -0.4721, -0.46921, -0.4674,
                          -0.46649, -0.46659, -0.46785, -0.47045, -0.4749,
                          -0.48081, -0.48822, -0.49726, -0.50805, -0.52086,
                          -0.53596, -0.55266, -0.57081, -0.59021, -0.61063,
                          -0.63238, -0.65484, -0.67722, -0.69907, -0.72004,
                          -0.73987, -0.75965, -0.77816, -0.79527, -0.8109,
                          -0.82509, -0.83844, -0.85159, -0.86373, -0.87474,
                          -0.88447, -0.8929, -0.90125, -0.90911, -0.91606,
                          -0.92197, -0.92677, -0.93071, -0.93528, -0.93931,
                          -0.9426, -0.94494, -0.94622, -0.94735, -0.94879,
                          -0.94965, -0.94971, -0.94878, -0.94679, -0.94571,
                          -0.94428, -0.94176, -0.93756, -0.9311, -0.92258,
                          -0.91342, -0.90142, -0.88573, -0.86549, -0.83998,
                          -0.81054, -0.77682, -0.73745, -0.692, -0.64033,
                          -0.58278, -0.52295, -0.45966, -0.39357, -0.32536,
                          -0.2557, -0.18627, -0.11888, -0.052529, 0.012911,
                          0.077684, 0.14195, 0.2039, 0.26391, 0.3225, 0.37976,
                          0.43566, 0.48977, 0.53999, 0.58745, 0.63232, 0.6749,
                          0.71548, 0.75342, 0.78841, 0.82157, 0.85356, 0.88519,
                          0.91734, 0.94972, 0.98403, 1.0219, 1.0648, 1.1143,
                          1.1713, 1.2359, 1.3092, 1.3911, 1.4809, 1.5776,
                          1.6787,
                          1.7819, 1.8846, 1.9842, 2.0777, 2.1627, 2.2362,
                          2.2969,
                          2.3436, 2.3759, 2.3942, 2.3994, 2.3927, 2.3763,
                          2.3521,
                          2.3222, 2.2886, 2.2524, 2.2149, 2.1772, 2.1394,
                          2.1018,
                          2.0641, 2.0259, 1.9878, 1.9494, 1.9106, 1.8711,
                          1.8305,
                          1.7886, 1.7455, 1.7008, 1.6537, 1.6036, 1.5495, 1.492,
                          1.431, 1.3664, 1.2981, 1.2262, 1.1508, 1.0736,
                          0.99504,
                          0.91575, 0.83626, 0.75657, 0.67816, 0.60206, 0.52843,
                          0.45712, 0.38776, 0.31881, 0.2515, 0.18511, 0.11894,
                          0.052471, -0.01492, -0.082568, -0.14848, -0.21193,
                          -0.27229, -0.32905, -0.38303, -0.43222, -0.47628,
                          -0.51587, -0.55188, -0.58534, -0.61763, -0.64653,
                          -0.67227, -0.69521, -0.71576, -0.73517, -0.75286,
                          -0.76798, -0.7812, -0.79326, -0.80474, -0.81719,
                          -0.82831, -0.83768, -0.84538, -0.85165, -0.85731,
                          -0.86309, -0.86791, -0.87271, -0.87846, -0.88592,
                          -0.89619, -0.90783, -0.91942, -0.93018, -0.93939]]])


class TestPyTorchConv1d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv1D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def test_FunctionForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        for convType in convTypes:
            print("Conv type: ", convType)
            conv = convType(filter_value=torch.from_numpy(y),
                            bias_value=torch.from_numpy(b))
            result = conv.forward(input=torch.from_numpy(x))
            np.testing.assert_array_almost_equal(
                result, np.array([[expected_result]]))

    def test_FunctionForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        for convType in convTypes:
            print("Conv type: ", convType)
            conv = convType(filter_value=torch.from_numpy(y),
                            bias_value=torch.from_numpy(b))
            result = conv.forward(input=torch.from_numpy(x))
            np.testing.assert_array_almost_equal(
                result, np.array([[expected_result]]))

    def test_FunctionForwardNoCompressionFFTForCompressSignalOnly(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardNoCompressionLongInput(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        for convType in convTypes:
            print("Conv type: ", convType)
            conv = convType(filter_value=torch.from_numpy(y),
                            bias_value=torch.from_numpy(b))
            result = conv.forward(input=torch.from_numpy(x))
            np.testing.assert_array_almost_equal(
                result, np.array([[expected_result]]))

    def test_FunctionForwardNoCompressionLongInputLength6(self):
        x = np.array([[[1., 2., 3., 1., 4., 5.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardNoCompressionLongInputLength7(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardWithCompressionLongInputLength7(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_full = np.correlate(x[0, 0, :], y[0, 0, :],
                                            mode="valid")
        print("\nexpected_result_full: ", expected_result_full)
        # expected_result_compressed = np.array(
        #     [14.59893, 31.640892, 46.518675, 24.199734, 26.882395])
        expected_result_compressed = np.array(
            [9.742255, 13.940483, 14.548998, 25.766991, 38.12536])

        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=5))
        result = conv.forward(input=torch.from_numpy(x))

        expected = np.array([[expected_result_compressed]])
        np.testing.assert_array_almost_equal(
            x=result, y=expected, decimal=6,
            err_msg="obtained x:{result} is different than "
                    "expected: {expected}!")

    def test_FunctionForwardWithCompressionPreserveEnergyLongInputLength7(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_full = np.correlate(x[0, 0, :], y[0, 0, :],
                                            mode="valid")
        print("\nexpected_result_full: ", expected_result_full)
        # expected_result_compressed = np.array(
        #     [14.59893, 31.640892, 46.518675, 24.199734, 26.882395])
        # expected_result_compressed = np.array(
        #     [[[42.426593, 46.477607, 35.418644, 31.677156, 0.]]])
        expected_result_compressed = np.array(
            [[[10.560706, 11.467013, 16.909018, 26.97858, 33.50754]]])

        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(preserve_energy=80))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result_compressed), y=result, decimal=6,
            err_msg=ERR_MSG)

    def test_FunctionForwardNoCompressionLongInputLength8(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10., 3.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardWithCompressionLongInputLength8(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10., 3.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_full = np.correlate(x[0, 0, :], y[0, 0, :],
                                            mode="valid")
        print("\nexpected_result_full: ", expected_result_full)
        # expected_result_compressed = np.array(
        #     [[[12.576667, 18.381238, 31.209274, 45.34337,
        #      28.775293, 21.985649]]])
        expected_result_compressed = np.array(
            [[[10.428204, 12.848331, 15.972131, 24.104296, 39.924989,
               31.94101]]])
        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=5))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            x=result, y=np.array(expected_result_compressed), decimal=6,
            err_msg="Obtained result x is different than expected y."
        )

    def test_FunctionForwardNoCompressionSimpleConvFFT(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardCompression(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.array(
            [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])

        for convType in [Conv1dfft]:
            print("Conv type: ", convType)
            conv = convType(filter_value=torch.from_numpy(y),
                            bias_value=torch.from_numpy(b),
                            args=Arguments(index_back=1))
            result = conv.forward(input=torch.from_numpy(x))
            np.testing.assert_array_almost_equal(
                x=np.array(expected_result), y=result,
                err_msg="Expected x is different from computed y.")

    def test_FunctionForwardCompressionConvFFTSimple1PercentIndexBack(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        print("expected_result_numpy: ", expected_result_numpy)
        # expected_result = np.array(
        #     [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        # expected_result = np.array([[[2.893933, 7.958111, 9.305407, 13.347296,
        #                               16.041889, 18.573978, 22.75877]]])
        expected_result = np.array(
            [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=1))
        result = conv.forward(input=torch.from_numpy(x))
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        result_tensor = tensor(result, dtype=torch.float32)
        print("absolute divergence: ",
              torch.sum(torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item())

    def test_FunctionForwardCompressionConvFFTSimple25PercentIndexBack(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        print("expected_result_numpy: ", expected_result_numpy)
        # expected_result = np.array(
        #     [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        # expected_result = np.array(
        #     [[[2.893933, 7.958111, 9.305407, 13.347296, 16.041889, 18.573978,
        #        22.75877]]]
        # )
        expected_result = np.array([[[1.893933, 6.958111, 11.305407,
                                      12.347296, 15.041889, 20.573978,
                                      21.75877]]])
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=25))
        result = conv.forward(input=torch.from_numpy(x))
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        result_tensor = tensor(result, dtype=torch.float32)
        print("absolute divergence 1 index back: ",
              torch.sum(torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item())

    def test_FunctionForwardCompressionConvFFTSimple25PercentIndexBack(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        print("expected_result_numpy: ", expected_result_numpy)
        # expected_result = np.array(
        #     [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        # expected_result = np.array(
        #     [[[2.893933, 7.958111, 9.305407, 13.347296, 16.041889, 18.573978,
        #        22.75877]]]
        # )
        expected_result = np.array([[[4.392661, 6.760233, 9.81906, 13.573233,
                                      15.376577, 19.196887, 22.567054]]])

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=25))
        result = conv.forward(input=torch.from_numpy(x))
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        result_tensor = tensor(result, dtype=torch.float32)
        print("absolute divergence 2 indexes back: ",
              torch.sum(torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item())

    def test_FunctionForwardCompressionConvFFTSimple50PercentIndexBack(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        print("expected_result_numpy: ", expected_result_numpy)
        # expected_result = np.array(
        #     [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        # expected_result = np.array(
        #     [[[2.893933, 7.958111, 9.305407, 13.347296, 16.041889, 18.573978,
        #        22.75877]]]
        # )
        # expected_result = np.array([[[1.893933, 6.958111, 11.305407,
        #                               12.347296, 15.041889, 20.573978,
        #                               21.75877]]])
        # expected_result = np.array(
        #     [[[4.056437, 4.361844, 8.24123, 13.879385, 18.638156,
        #        20.290859, 18.064178]]])
        expected_result = np.array(
            [[[4.642227, 7.571062, 9.331596, 11.987442, 16.695069, 20.921597,
               20.311209]]])

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=50))
        result = conv.forward(input=torch.from_numpy(x))
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        result_tensor = tensor(result, dtype=torch.float32)
        print("absolute divergence 3 indexes back: ",
              torch.sum(torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item())

    def test_FunctionForwardCompressionConvFFTSimple70PercentIndexBack(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        print("expected_result_numpy: ", expected_result_numpy)
        # expected_result = np.array(
        #     [[[4.25, 6.75, 10.25, 12.75, 16.25, 18.75, 22.25]]])
        # expected_result = np.array(
        #     [[[2.893933, 7.958111, 9.305407, 13.347296, 16.041889, 18.573978,
        #        22.75877]]]
        # )
        # expected_result = np.array([[[1.893933, 6.958111, 11.305407,
        #                               12.347296, 15.041889, 20.573978,
        #                               21.75877]]])
        # expected_result = np.array(
        #     [[[12., 12., 12., 12., 12., 12., 12.]]])
        expected_result = np.array(
            [[[4.978448, 8.778, 12.268808, 14.919427, 16.326325, 16.275314,
               14.774161]]])

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=70))
        result = conv.forward(input=torch.from_numpy(x))
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        result_tensor = tensor(result, dtype=torch.float32)
        print("absolute divergence 4 indexes back: ",
              torch.sum(torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item())

    def test_FunctionForwardCompressionConvFFTPreserveEnergy(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 90.,
        #                       80., 10.]
        preserved_energies = [95.]

        for preserve_energy in preserved_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            is_debug=True,
                                            compress_type=CompressType.NO_FILTER))
            result = conv.forward(input=torch.from_numpy(x))
            print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print("absolute divergence for preserved energy {} is {}".format(
                preserve_energy, torch.sum(
                    torch.abs(result_tensor - expected_result_tensor),
                    dim=-1).item()))

    def test_FunctionForwardCompressionConvFFTPreserveEnergy50wordsBigCoeff(
            self):
        x = fifty_words
        print("length of the input signal: ", x.shape[-1])
        y = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = stats.zscore(y)
        y = np.array([[y]])
        print("length of the filter: ", y.shape[-1])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        # print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 85.,
        #                       80., 70., 60., 50., 40., 10.]
        preserved_energies = [95.]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        for preserve_energy in preserved_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            is_debug=True,
                                            next_power2=False,
                                            compress_type=CompressType.BIG_COEFF))
            result = conv.forward(input=torch.from_numpy(x))
            # print("actual result: ", result)

            result = result.float()
            print(
                "absolute divergence for preserved energy,{},is,{},stop".format(
                    preserve_energy, torch.sum(
                        torch.abs(result - expected_result_tensor),
                        dim=-1).item()))

    def test_FunctionForwardCompressionConvFFTPreserveEnergy50wordsLowCoeff(
            self):
        x = fifty_words
        print("length of the input signal: ", x.shape[-1])
        y = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = stats.zscore(y)
        y = np.array([[y]])
        print("length of the filter: ", y.shape[-1])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        # print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 85.,
        #                      80., 70., 60., 50., 40., 10.]
        preserved_energies = [95.]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        for preserve_energy in preserved_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            is_debug=True, next_power2=False,
                                            compress_type=CompressType.LOW_COEFF))
            result = conv.forward(input=torch.from_numpy(x))
            # print("actual result: ", result)

            result = result.float()
            print(
                "absolute divergence for preserved energy,{},is,{},stop".format(
                    preserve_energy, torch.sum(
                        torch.abs(result - expected_result_tensor),
                        dim=-1).item()))

    def test_FunctionForwardCompressionConvFFTIndexBack50words(self):
        x = fifty_words
        print("length of the input signal: ", x.shape[-1])
        y = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = stats.zscore(y)
        y = np.array([[y]])
        print("length of the filter: ", y.shape[-1])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        # print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 85.,
        #                        80., 70., 60., 50., 40., 10.]
        # preserved_energies = [95.]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 100]  # percent
        indexes_back = [8]  # percent

        for index_back in indexes_back:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(index_back=index_back,
                                            preserve_energy=None,
                                            is_debug=True,
                                            compress_type=CompressType.LOW_COEFF,
                                            next_power2=False))
            result = conv.forward(input=torch.from_numpy(x))
            # print("actual result: ", result)

            result_tensor = result.float()
            print(
                "absolute divergence for index back,{},is,{},stop".format(
                    index_back, torch.sum(
                        torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item()))

    def test_FunctionForwardCompressionConvFFTIndexBackBigCoeff50words(self):
        x = fifty_words
        print("length of the input signal: ", x.shape[-1])
        y = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = stats.zscore(y)
        y = np.array([[y]])
        print("length of the filter: ", y.shape[-1])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        # print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 85.,
        #                        80., 70., 60., 50., 40., 10.]
        # preserved_energies = [95.]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 100]  # percent
        indexes_back = [8]

        for index_back in indexes_back:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(index_back=index_back,
                                            preserve_energy=None,
                                            is_debug=True,
                                            next_power2=False,
                                            compress_type=CompressType.BIG_COEFF))
            result = conv.forward(input=torch.from_numpy(x))
            # print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print(
                "absolute divergence for index back,{},is,{},stop".format(
                    index_back, torch.sum(
                        torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item()))

    def test_FunctionForwardCompressionConvFFTPreserveEnergyAdiac(self):
        x = np.array([[[1.598, 1.5994, 1.5705, 1.5505, 1.5074, 1.4343, 1.369,
                        1.3053, 1.2103, 1.1167, 1.024, 0.92598, 0.82811,
                        0.73922, 0.64305, 0.55643, 0.46295, 0.36958, 0.27843,
                        0.18583, 0.095532, 0.010646, -0.080856, -0.16555,
                        -0.24349, -0.33543, -0.42611, -0.49749, -0.56685,
                        -0.64986, -0.73225, -0.7778, -0.84298, -0.9173,
                        -0.98797, -1.055, -1.0968, -1.1471, -1.1984, -1.2432,
                        -1.2816, -1.3126, -1.3357, -1.3511, -1.3769, -1.3879,
                        -1.3769, -1.3571, -1.3296, -1.2937, -1.2577, -1.2304,
                        -1.1742, -1.1142, -1.072, -1.0068, -0.94893, -0.87017,
                        -0.79017, -0.72451, -0.64284, -0.59163, -0.52139,
                        -0.43063, -0.35085, -0.26986, -0.17364, -0.089113,
                        -0.0039203, 0.085453, 0.17518, 0.26706, 0.36297,
                        0.45238, 0.54911, 0.64982, 0.74624, 0.84566, 0.94633,
                        1.0457, 1.1463, 1.2424, 1.344, 1.4418, 1.5407, 1.6241,
                        1.6696, 1.6852, 1.6967, 1.6975, 1.6874, 1.6385, 1.5737,
                        1.4786, 1.3769, 1.2766, 1.177, 1.0777, 0.9792, 0.88053,
                        0.77793, 0.68164, 0.59043, 0.49275, 0.40434, 0.30859,
                        0.21352, 0.12268, 0.037603, -0.052602, -0.13751,
                        -0.2204, -0.31556, -0.39461, -0.44608, -0.53344,
                        -0.60286, -0.67341, -0.73761, -0.81903, -0.88526,
                        -0.95008, -1.0026, -1.063, -1.1252, -1.1658, -1.2043,
                        -1.2485, -1.2738, -1.2844, -1.2733, -1.2867, -1.2919,
                        -1.2886, -1.2777, -1.2581, -1.2314, -1.1973, -1.1563,
                        -1.1105, -1.0845, -1.0258, -0.96255, -0.89507, -0.84781,
                        -0.77254, -0.69457, -0.632, -0.55122, -0.48459,
                        -0.40868, -0.32442, -0.24745, -0.16365, -0.076913,
                        0.0088012, 0.097274, 0.1859, 0.284, 0.38121, 0.46303,
                        0.55693, 0.65097, 0.7382, 0.83474, 0.92994, 1.0249,
                        1.1209, 1.2172, 1.3125, 1.4029, 1.481, 1.521, 1.5642,
                        1.5709, 1.5929]]])
        print("length of the input signal: ", x.shape[-1])
        y = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = stats.zscore(y)
        y = np.array([[y]])
        print("length of the filter: ", y.shape[-1])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        # print("expected_result_numpy: ", expected_result_numpy)

        # preserved_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 85.,
        #                       80., 70., 60., 50., 40., 10.]
        preserved_energies = [50.]

        for preserve_energy in preserved_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            is_debug=True, next_power2=False,
                                            compress_type=CompressType.STANDARD))
            result = conv.forward(input=torch.from_numpy(x))
            # print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print(
                "absolute divergence for preserved energy,{},is,{},stop".format(
                    preserve_energy, torch.sum(
                        torch.abs(result_tensor - expected_result_tensor),
                        dim=-1).item()))

    def test_FunctionForwardCompressionSignalOnlyPreserveEnergy(self):
        # x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
        # 13., 14., 15., 16., 17., 18., 19., 20.]]])
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        print("expected_result_numpy: ", expected_result_numpy)

        # preserve_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 90., 80.,
        #                      10., 1.]
        preserve_energies = [50.0]
        for preserve_energy in preserve_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            next_power2=False))
            result = conv.forward(input=torch.from_numpy(x))
            print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print("absolute divergence for preserved energy {} is {}".format(
                preserve_energy, torch.sum(
                    torch.abs(result_tensor - expected_result_tensor),
                    dim=-1).item()))

    def test_FunctionForwardCompressionConvSimpleForLoopPreserveEnergy(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        print("expected_result_numpy: ", expected_result_numpy)

        # preserve_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 90.,
        #                      80., 10.]
        preserve_energies = [50.0]
        for preserve_energy in preserve_energies:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             args=Arguments(preserve_energy=preserve_energy,
                                            next_power2=True))
            result = conv.forward(input=torch.from_numpy(x))
            print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print("absolute divergence for preserved energy {} is {}".format(
                preserve_energy, torch.sum(
                    torch.abs(result_tensor - expected_result_tensor),
                    dim=-1).item()))

    def test_FunctionForwardSpectralPooling(self):
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        # expected_result = np.array(
        #     [[[2.771341, 5.15668, 9.354594, 14.419427]]])
        expected_result = np.array(
            [[[4.904259, 6.398284, 8.617751, 12.688923]]])
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         out_size=4)
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result,
            err_msg="Expected x is different from computed y.")

    def test_FunctionForwardNoCompressionManySignalsOneChannel(self):
        x = np.array([[[1., -1., 0.]], [[1., 2., 3.]]])
        y = np.array([[[-2.0, 3.0]]])
        b = np.array([0.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompressionManySignalsOneChannelConvSimpleFFT(
            self):
        x = np.array([[[1., -1., 0.]], [[1., 2., 3.]]])
        y = np.array([[[-2.0, 3.0]]])
        b = np.array([0.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompressionManySignalsOneFilterTwoChannels(self):
        x = np.array([[[1., 2., 3.], [4., 5., 6.]],
                      [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([0.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompressionManySignalsOneFilterTwoChannelsSimpleFFTConv(
            self):
        x = np.array([[[1., 2., 3.], [4., 5., 6.]],
                      [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([1.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompression2Signals2Filters2Channels(self):
        x = np.array(
            [[[1., 2., 3.], [4., 5., 6.]], [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[2., 1.], [1., 3.]], [[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([1.0, 2.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompression2Signals2Filters2ChannelsSipleFFTConv(
            self):
        x = np.array(
            [[[1., 2., 3.], [4., 5., 6.]], [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[2., 1.], [1., 3.]], [[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([1.0, 2.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardRandom(self):
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardRandomSipleFFTConv(self):
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))

        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardRandomFFTSimpleCompressSignalOnly(self):
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionBackwardNoCompressionWithBias(self):
        x = np.array([[[1.0, 2.0, 3.0]]])
        y = np.array([[[2.0, 1.0]]])
        b = np.array([2.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b,
                                                       conv_param)
        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = Conv1dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _, _ = Conv1dfftFunction.backward(
            ctx,
            dout)

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(dx))

        # are the gradients correct
        np.testing.assert_array_almost_equal(dx.detach().numpy(),
                                             expected_dx)
        np.testing.assert_array_almost_equal(dw.detach().numpy(),
                                             expected_dw)
        np.testing.assert_array_almost_equal(db.detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionWithBias2Inputs(self):
        x = np.array([[[1.0, 2.0, 3.0]], [[2.0, -1.0, 3.0]]])
        # Number of filters F, number of channels C, number of values WW.
        y = np.array([[[2.0, 1.0]]])
        b = np.array([2.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)
        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = Conv1dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]], [[0.2, -0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        print("expected_dx: ", expected_dx)
        print("expected_dw: ", expected_dw)
        print("expected_db: ", expected_db)

        dx, dw, db, _, _, _, _, _, _ = Conv1dfftFunction.backward(
            ctx, dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=dx.detach().numpy(),
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=dw.detach().numpy(),
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(db.detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionWithBias2Filters(self):
        x = np.array([[[1.0, 2.0, 3.0]]])
        # Number of filters F, number of channels C, number of values WW.
        y = np.array([[[2.0, 1.0]], [[-0.1, 0.3]]])
        b = np.array([2.0, 1.0])  # 2 filters => 2 bias terms
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)
        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = Conv1dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2], [0.2, -0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        print("expected_dx: ", expected_dx)
        print("expected_dw: ", expected_dw)
        print("expected_db: ", expected_db)

        dx, dw, db, _, _, _, _, _, _ = Conv1dfftFunction.backward(
            ctx,
            dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=dx.detach().numpy(),
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=dw.detach().numpy(),
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(db.detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionNoBias(self):
        print()
        print("Test forward and backward manual passes.")
        x = np.array([[[1.0, 2.0, 3.0]]])
        y = np.array([[[2.0, 1.0]]])
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)

        conv1dfft = Conv1dfft(filter_value=y_torch, bias_value=b_torch)
        result_torch = conv1dfft.forward(input=x_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(x_torch.grad,
                                             expected_dx)
        np.testing.assert_array_almost_equal(y_torch.grad,
                                             expected_dw)
        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

    def test_FunctionBackwardNoCompressionNoBiasLonger(self):
        print()
        print("Test forward and backward manual passes.")
        x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 3.0]]])
        y = np.array([[[2.0, 1.0, -2.0]]])
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)

        print("expected result out for the forward pass: ", expected_result)

        result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2, 0.3, -0.1, 0.4]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        print("expected_dx: ", expected_dx)
        print("expected_dw: ", expected_dw)
        print("expected_db: ", expected_db)

        result_torch.backward(dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(x=expected_dx, y=x_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_dw, y=y_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_db, y=b_torch.grad,
                                             err_msg=ERR_MSG)

    def test_FunctionBackwardWithPooling(self):
        x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -2.0]]])
        y = np.array([[[2.0, 1.0, 3.0, 1.0, -3.0]]])
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        accurate_expected_result, cache = conv_forward_naive_1D(
            x=x, w=y, b=b, conv_param=conv_param)
        print("Accurate expected result: ", accurate_expected_result)

        # approximate_expected_result = np.array(
        #     [[[-2.105834, 0.457627, 8.501472, 20.74531]]])
        # approximate_expected_result = np.array(
        #     [[[6.146684, 11.792807, 17.264324, 21.90055]]])
        approximate_expected_result = np.array(
            [[[-3.035742, 4.273538, 14.162642, 24.351469]]])

        print("Approximate expected result: ", approximate_expected_result)

        out_size = approximate_expected_result.shape[-1]

        conv1dfft = Conv1dfft(out_size=out_size, filter_value=y_torch,
                              bias_value=b_torch, args=Arguments(
                compress_type=CompressType.STANDARD))
        result_torch = conv1dfft.forward(input=x_torch)
        result = result_torch.detach().numpy()
        print("Computed result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(approximate_expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        self._check_delta1D(actual_result=result,
                            accurate_expected_result=accurate_expected_result,
                            delta=11.1)

        dout = tensor([[[0.1, -0.2, 0.3, -0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # approximate_expected_dx = np.array(
        #     [[[0.052956, 0.120672, 0.161284, 0.150332, 0.089258,
        #        0.005318, -0.063087, -0.087266, -0.063311, -0.012829]]])
        approximate_expected_dx = np.array(
            [[[0.086085, 0.11464, 0.122151, 0.105543, 0.069081, 0.023023,
               -0.019728, -0.047839, -0.055165, -0.042567]]])

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=approximate_expected_dx, y=x_torch.grad,
            err_msg="Expected x is different from computed y.")

        self._check_delta1D(actual_result=x_torch.grad,
                            accurate_expected_result=expected_dx, delta=1.1)

        # approximate_expected_dw = np.array(
        #     [[[0.129913, 0.249468, 0.429712, 0.620098, 0.748242]]])
        approximate_expected_dw = np.array(
            [[[0.162962, 0.323346, 0.492451, 0.629942, 0.7004]]])
        np.testing.assert_array_almost_equal(
            x=approximate_expected_dw, y=y_torch.grad,
            err_msg="Expected x is different from computed y.")

        self._check_delta1D(actual_result=y_torch.grad,
                            accurate_expected_result=expected_dw, delta=0.2)

        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

    def _check_delta1D(self, actual_result, accurate_expected_result, delta):
        """
        Compare if the difference between the two objects is more than the
        given delta.

        :param actual_result: the computed result
        :param accurate_expected_result: the expected accurate result
        :param delta: compare if that the difference between the two objects
        is more than the given delta
        """
        print("actual_result: {}".format(actual_result))
        print("accurate_expected_result: {}".format(accurate_expected_result))
        result_flat = actual_result[0][0]
        accurate_expected_flat = accurate_expected_result[0][0]
        for index, item in enumerate(result_flat):
            self.assertAlmostEqual(
                first=accurate_expected_flat[index], second=item, delta=delta,
                msg="The approximate result is not within delta={} of the "
                    "accurate result!".format(delta))

    def test_FunctionBackwardCompressionBias(self):
        x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 10.0]]])
        y = np.array([[[2.0, 1.0, -3.0]]])
        b = np.array([1.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)
        print("expected result: ", expected_result)

        # 1 index back
        conv_fft = Conv1dfft(filter_value=y_torch, bias_value=b_torch,
                             args=Arguments(index_back=1, preserve_energy=None))
        result_torch = conv_fft.forward(input=x_torch)

        result = result_torch.detach().numpy()
        # compressed_expected_result = np.array(
        #     [[[-2.25, -5.75, -2.25, 15.25, -18.25]]])
        # compressed_expected_result = np.array(
        #     [[[-4., -3.999999, -4., 16.999998, -20.]]])
        compressed_expected_result = np.array(
            [[[-0.35, -7.95, 0.08, 12.97, -16.2]]])
        np.testing.assert_array_almost_equal(
            x=compressed_expected_result, y=result, decimal=2, err_msg=ERR_MSG)

        dout = tensor([[[0.1, -0.2, -0.3, 0.3, 0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)
        assert conv_fft.is_manual[0] == 1

        # are the gradients correct
        print("accurate expected_dx: ", expected_dx)
        # approximate_dx = np.array(
        #     [[[0.175, -0.275, -1.125, 0.925, 1.375, -0.775, -0.325]]])
        # approximate_dx = np.array(
        #     [[[0.2, -0.3, -1.1, 0.9, 1.4, -0.8, -0.3]]])
        approximate_dx = np.array(
            [[[0.199, -0.285, -1.13, 0.942, 1.347, -0.738, -0.368]]])
        np.testing.assert_array_almost_equal(
            x=approximate_dx, y=x_torch.grad, decimal=3,
            err_msg="Expected approximate x is different from computed y. The "
                    "exact x (that represents dx) is: {}".format(expected_dx))
        print("accurate expected_dw: ", expected_dw)

        # approximate_dw = np.array([[[0.675, -0.375, -1.125]]])
        # approximate_dw = np.array([[[0.5, -0.2, -1.3]]])
        approximate_dw = np.array([[[0.918, -0.635, -0.867]]])

        np.testing.assert_array_almost_equal(
            x=approximate_dw, y=y_torch.grad, decimal=3,
            err_msg="Expected approximate x is different from computed y. The "
                    "exact x (that represents dw) is: {}".format(expected_dw))
        np.testing.assert_array_almost_equal(
            x=expected_db, y=b_torch.grad,
            err_msg="Expected approximate x is different from computed y.")

    def test_FunctionBackwardNoCompression2Channels(self):
        x = np.array([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
        y = np.array([[[2.0, 1.0], [-2.0, 3.0]]])
        # still it is only a single filter but with 2 channels
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)
        result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch)

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        print()
        print("expected dx: " + str(expected_dx))
        print("computed dx: {}".format(x_torch.grad))

        print("expected dw: {}".format(expected_dw))
        print("computed dw: {}".format(y_torch.grad))

        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # Are the gradients correct?
        np.testing.assert_array_almost_equal(x=expected_dx, y=x_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_dw, y=y_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_db, y=b_torch.grad,
                                             err_msg=ERR_MSG)

    def test_FunctionBackwardNoCompression2ChannelsWithPadding(self):
        x = np.array([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
        y = np.array([[[2.0, 1.0], [-2.0, 3.0]]])
        # still it is only a single filter but with 2 channels
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        padding = 1

        conv_param = {'pad': padding, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)

        result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch,
                                               padding)

        # out_W = W + 2 * pad - WW + 1

        dout = tensor([[[0.1, -0.2, 0.3, -0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, )

        print()
        print("expected dx: " + str(expected_dx))
        print("computed dx: {}".format(x_torch.grad))

        print("expected dw: {}".format(expected_dw))
        print("computed dw: {}".format(y_torch.grad))

        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # Are the gradients correct?
        np.testing.assert_array_almost_equal(x=expected_dx, y=x_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_dw, y=y_torch.grad,
                                             err_msg=ERR_MSG)
        np.testing.assert_array_almost_equal(x=expected_db, y=b_torch.grad,
                                             err_msg=ERR_MSG)

    def test_FunctionForwardWithCompression(self):
        # test with compression
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # expected_result = [[[3.75, 7.25]]]
        expected_result = [[[3.666667, 7.333333]]]
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b),
                         args=Arguments(index_back=1))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

    def test_AutogradForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfft(filter_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))

        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_AutogradForwardWithCompression(self):
        # test with compression
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        expected_result = [3.666667, 7.333333]
        conv = Conv1dfft(
            filter_value=torch.from_numpy(y), bias_value=torch.from_numpy(b),
            args=Arguments(index_back=1, preserve_energy=None))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardBackwardRandomAutoGrad(self):
        print()
        print("Test forward backward passes with random data.")
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 6
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x=x, w=y, b=b,
                                                   conv_param=conv_param)
        # self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)
        conv = Conv1dfft(filter_value=y_torch, bias_value=b_torch)
        result_torch = conv.forward(input=x_torch)
        result = result_torch.detach().numpy()
        # self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)

        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # Assert that we executed the backward pass via PyTorch's AutoGrad
        # (value is 0) and not manually (for manual grad the value is 1:
        # conv.is_manual[0] == 1).
        assert 1 == conv.is_manual[0]

        # print()
        # print("expected dx: " + str(expected_dx))
        # print("computed dx: {}".format(x_torch.grad))
        #
        # print("expected dw: {}".format(expected_dw))
        # print("computed dw: {}".format(y_torch.grad))
        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=x_torch.grad, decimal=5,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=y_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_db, y=b_torch.grad, decimal=5,
            err_msg="Expected x is different from computed y.")

    def test_FunctionForwardBackwardRandomManualBackprop(self):
        print()
        print("Test forward backward manual passes with random data.")
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 5
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x=x, w=y, b=b,
                                                   conv_param=conv_param)
        # self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)
        conv = Conv1dfft(filter_value=y_torch, bias_value=b_torch)
        result_torch = conv.forward(input=x_torch)

        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape, dtype=dtype, device=device)

        result_torch.backward(dout)

        result = result_torch.cpu().detach().numpy()
        # self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            np.array(expected_result), result, decimal=5)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.cpu().numpy(), cache)

        # Assert that we executed the backward pass manually (value is 1) and
        # not via PyTorch's autograd (for autograd: conv.is_manual[0] == 0).
        assert 1 == conv.is_manual[0]

        # print()
        # print("expected dx: " + str(expected_dx))
        # print("computed dx: {}".format(x_torch.grad))
        #
        # print("expected dw: {}".format(expected_dw))
        # print("computed dw: {}".format(y_torch.grad))
        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=x_torch.grad.cpu().detach().numpy(),
            decimal=5,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=y_torch.grad.cpu().detach().numpy(),
            decimal=4,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_db, y=b_torch.grad.cpu().detach().numpy(),
            decimal=5,
            err_msg="Expected x is different from computed y.")

    def test_FunctionForwardBackwardCountTensors(self):
        print()
        print("Test forward manual pass and counting tensors within ctx.")

        clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        only_cuda = True if torch.cuda.is_available() else False

        x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 3.0]]])
        y = np.array([[[2.0, 1.0, -2.0]]])
        b = np.array([0.0])
        dtype = torch.float32
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x=x, w=y, b=b,
                                                       conv_param=conv_param)

        print("expected result out for the forward pass: ", expected_result)
        conv = Conv1dfft(filter_value=y_torch, bias_value=b_torch)

        result_torch = conv.forward(x_torch)

        print("Get all tensors visible in the test case: ")
        tensors = get_tensors(only_cuda=False)
        print("tensors: ", ",".join([str(tensor) for tensor in tensors]))

        """The forward pass saves 11 tensors for backward pass,the local code 
        in the method created 3 tensors, the output result_torch is another 
        tensor, which gives 15 tensors in total. Additionally, the device,
        dtype and the input to Conv1dfft constructor is the tensor is_manual."""
        expect = 15 + 3
        if len(tensors) != expect:
            raise Exception(f"Expected {expect} tensors but got {len(tensors)}")

        result = result_torch.cpu().detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = torch.tensor([[[0.1, -0.2, 0.1, -0.1, 0.3]]], dtype=dtype,
                            device=device)

        result_torch.backward(dout)

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.cpu().numpy(), cache)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x_torch.grad.cpu().detach().numpy(),
            expected_dx)
        np.testing.assert_array_almost_equal(
            y_torch.grad.cpu().detach().numpy(),
            expected_dw)
        np.testing.assert_array_almost_equal(
            b_torch.grad.cpu().detach().numpy(),
            expected_db)


if __name__ == '__main__':
    unittest.main()
