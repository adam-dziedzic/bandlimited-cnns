from __future__ import division
from numba import cuda
import numpy as np
import math
import torch
import os
import ctypes

#os.environ['NUMBAPRO_LIBDEVICE']='/usr/lib/nvidia-cuda-toolkit/libdevice/'
#os.environ['NUMBAPRO_NVVM']='/usr/lib/x86_64-linux-gnu/libnvvm.so.3.1.0'

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2  # do the computation


def get_devicendarray(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'),
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)

# Host code
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

#data = np.ones(256)
#data_gpu = cuda.to_device(data)
#data_torch = torch.from_numpy(get_devicendarray(data_gpu))
data_torch = torch.ones(256, device=device)
threadsperblock = 256
blockspergrid = math.ceil(data_torch.size()[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](get_devicendarray(data_torch))
print(data_torch)
