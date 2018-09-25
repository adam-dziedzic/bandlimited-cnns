import logging
import unittest

import numpy as np
import torch
from torch import tensor

from cnns.nnlib.layers import conv_backward_naive_1D
from cnns.nnlib.layers import conv_forward_naive_1D
from cnns.nnlib.pytorch_layers.conv1D_fft \
    import Conv1dfftFunction, Conv1dfftAutograd, Conv1dfft, Conv1dfftSimple, \
    Conv1dfftCompressSignalOnly, Conv1dfftSimpleForLoop
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging

ERR_MSG = "Expected x is different from computed y."

convTypes = [Conv1dfftAutograd, Conv1dfft, Conv1dfftSimple,
             Conv1dfftSimpleForLoop, Conv1dfftCompressSignalOnly]


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

    def test_FunctionForwardNoCompressionFFTForCompressSignalOnly(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
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
        expected_result_compressed = np.array(
            [14.59893, 31.640892, 46.518675, 24.199734, 26.882395])
        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
                                           bias_value=torch.from_numpy(b),
                                           index_back=5)
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result_compressed]]), decimal=6)

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
        expected_result_compressed = np.array(
            [42.426593, 46.477607, 35.418644, 31.677156, 0.])
        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
                                           bias_value=torch.from_numpy(b),
                                           preserve_energy=80)
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            x=np.array([[expected_result_compressed]]), y=result, decimal=6,
            err_msg=ERR_MSG)

    def test_FunctionForwardNoCompressionLongInputLength8(self):
        x = np.array([[[1., 2., 3., 1., 4., 5., 10., 3.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
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
        expected_result_compressed = np.array(
            [12.576667, 18.381238, 31.209274, 45.34337,
             28.775293, 21.985649])
        print("\nexpected_result_compressed: ", expected_result_compressed)

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
                                           bias_value=torch.from_numpy(b),
                                           index_back=5)
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result_compressed]]), decimal=6)

    def test_FunctionForwardNoCompressionSimpleConvFFT(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
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
                            bias_value=torch.from_numpy(b), index_back=1)
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
        expected_result = np.array([[[2.893933, 7.958111, 9.305407, 13.347296,
                                      16.041889, 18.573978, 22.75877]]])
        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
                               bias_value=torch.from_numpy(b), index_back=1)
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
        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
                               bias_value=torch.from_numpy(b), index_back=25)
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
        expected_result = np.array([[[1.893933, 6.958111, 11.305407,
                                      12.347296, 15.041889, 20.573978,
                                      21.75877]]])
        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
                               bias_value=torch.from_numpy(b), index_back=25)
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
        expected_result = np.array(
            [[[4.056437, 4.361844, 8.24123, 13.879385, 18.638156,
               20.290859, 18.064178]]])

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
                               bias_value=torch.from_numpy(b), index_back=50)
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
        expected_result = np.array(
            [[[12., 12., 12., 12., 12., 12., 12.]]])

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
                               bias_value=torch.from_numpy(b), index_back=70)
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

        for preserve_energy in [100., 99.5, 99.1, 99.0, 97., 96., 95., 90.,
                                80., 10.]:
            conv = Conv1dfft(filter_value=torch.from_numpy(y),
                             bias_value=torch.from_numpy(b),
                             preserve_energy=preserve_energy)
            result = conv.forward(input=torch.from_numpy(x))
            print("actual result: ", result)

            result_tensor = tensor(result, dtype=torch.float32)
            print("absolute divergence for preserved energy {} is {}".format(
                preserve_energy, torch.sum(
                    torch.abs(result_tensor - expected_result_tensor),
                    dim=-1).item()))

    def test_FunctionForwardCompressionSignalOnlyPreserveEnergy(self):
        # x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
        #                 14., 15., 16., 17., 18., 19., 20.]]])
        x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result_numpy = np.correlate(x[0, 0, :], y[0, 0, :],
                                             mode="valid")
        expected_result_tensor = tensor([[expected_result_numpy]],
                                        dtype=torch.float32)
        print("expected_result_numpy: ", expected_result_numpy)

        preserve_energies = [100., 99.5, 99.1, 99.0, 97., 96., 95., 90., 80.,
                             10., 1.]
        # preserve_energies = [50.0]
        for preserve_energy in preserve_energies:
            conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
                                               bias_value=torch.from_numpy(b),
                                               preserve_energy=preserve_energy,
                                               use_next_power2=True)
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
            conv = Conv1dfftSimpleForLoop(filter_value=torch.from_numpy(y),
                                          bias_value=torch.from_numpy(b),
                                          preserve_energy=preserve_energy,
                                          use_next_power2=True,
                                          is_complex_pad=True)
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
                         bias_value=torch.from_numpy(b), out_size=4)
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

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
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

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
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

        dx, dw, db, _, _, _, _, _, _, _, _, _ = Conv1dfftFunction.backward(ctx,
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

        dx, dw, db, _, _, _, _, _, _, _, _, _ = Conv1dfftFunction.backward(
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

        dx, dw, db, _, _, _, _, _, _, _, _, _ = Conv1dfftFunction.backward(ctx,
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

        result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch)
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
        approximate_expected_result = np.array(
            [[[6.146684, 11.792807, 17.264324, 21.90055]]])
        print("Approximate expected result: ", approximate_expected_result)

        out_size = approximate_expected_result.shape[-1]

        result_torch = Conv1dfftFunction.apply(
            x_torch, y_torch, b_torch, None, None, None, None, out_size, 1,
            True)
        result = result_torch.detach().numpy()
        print("Computed result: ", result)
        np.testing.assert_array_almost_equal(
            x=np.array(approximate_expected_result), y=result,
            err_msg="Expected x is different from computed y.")

        self._check_delta1D(actual_result=result,
                            accurate_expected_result=accurate_expected_result,
                            delta=8.1)

        dout = tensor([[[0.1, -0.2, 0.3, -0.1]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # approximate_expected_dx = np.array(
        #     [[[0.052956, 0.120672, 0.161284, 0.150332, 0.089258,
        #        0.005318, -0.063087, -0.087266, -0.063311, -0.012829]]])
        approximate_expected_dx = np.array(
            [[[0.069143, 0.073756, 0.072217, 0.064673, 0.052066,
               0.035999, 0.018504, 0.001745, -0.012286, -0.022057]]])

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=approximate_expected_dx, y=x_torch.grad,
            err_msg="Expected x is different from computed y.")

        self._check_delta1D(actual_result=x_torch.grad,
                            accurate_expected_result=expected_dx, delta=1.1)

        # approximate_expected_dw = np.array(
        #     [[[0.129913, 0.249468, 0.429712, 0.620098, 0.748242]]])
        approximate_expected_dw = np.array(
            [[[0.287457, 0.391437, 0.479054, 0.539719, 0.565961]]])
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
                             index_back=1)
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
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

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
        conv = Conv1dfftFunction()
        result = conv.forward(
            ctx=None, input=torch.from_numpy(x), index_back=1,
            filter=torch.from_numpy(y), bias=torch.from_numpy(b))
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

    def test_AutogradForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        conv = Conv1dfftAutograd(filter_value=torch.from_numpy(y),
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
        conv = Conv1dfftAutograd(
            filter_value=torch.from_numpy(y), bias_value=torch.from_numpy(b),
            index_back=1)
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
        num_filters = 3
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
        self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)
        conv = Conv1dfftAutograd(filter_value=y_torch, bias_value=b_torch,
                                 is_manual=tensor([0]))
        result_torch = conv.forward(input=x_torch)
        result = result_torch.detach().numpy()
        self.logger.debug("obtained result: " + str(result))
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
        assert 0 == conv.is_manual[0]

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
        self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)
        conv = Conv1dfft(filter_value=y_torch, bias_value=b_torch)
        result_torch = conv.forward(input=x_torch)
        result = result_torch.detach().numpy()
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)

        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # Assert that we executed the backward pass manually (value is 1) and
        # not via PyTorch's autograd (for autograd: conv.is_manual[0] == 0).
        assert conv.is_manual[0] == 1

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


if __name__ == '__main__':
    unittest.main()
