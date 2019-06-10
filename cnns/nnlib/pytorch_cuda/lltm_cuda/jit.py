from torch.utils.cpp_extension import load

lltm_cuda = load(
    'lltm_cuda', ['lltm_cuda.lltm_cpp', 'band_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)
