import torch
import time
from cnns.nnlib.pytorch_cuda.cpp.lltm import LLTM as LLTM_CPP
from cnns.nnlib.pytorch_cuda.python.lltm import LLTM as LLTM_PY
from cnns.nnlib.pytorch_cuda.cuda.lltm import LLTM as LLTM_CUDA

batch_size = 16
input_features = 32
state_size = 128

device = torch.device("cpu")


def init_vars(device):
    X = torch.randn(batch_size, input_features, device=device)
    h = torch.randn(batch_size, state_size, device=device)
    C = torch.randn(batch_size, state_size, device=device)

    return X, h, C


def test(X, h, C, device=torch.device("cpu")):
    forward = 0
    backward = 0
    for _ in range(100000):  # default: 100000
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        if device is torch.device("cuda"):
            torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        if device is torch.device("cuda"):
            torch.cuda.synchronize()
        backward += time.time() - start

    return forward, backward


FORWARD_BACK_INFO = "device,{},mode,{},Forward,{:.3f},sec,Backward,{:.3f},sec"

device = torch.device("cpu")

X, h, C = init_vars(device=device)
rnn = LLTM_PY(input_features, state_size, device=device)
forward, backward = test(X, h, C, device=device)
print(FORWARD_BACK_INFO.format(str(device), 'python', forward, backward))

X, h, C = init_vars(device=device)
rnn = LLTM_CPP(input_features, state_size, device=device)
forward, backward = test(X, h, C, device=device)
print(FORWARD_BACK_INFO.format(str(device), 'cpp', forward, backward))

device = torch.device("cuda")

X, h, C = init_vars(device=device)
rnn = LLTM_PY(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print(FORWARD_BACK_INFO.format(str(device), 'python', forward, backward))

X, h, C = init_vars(device=device)
rnn = LLTM_CPP(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print(FORWARD_BACK_INFO.format(str(device), 'cpp', forward, backward))

X, h, C = init_vars(device=device)
rnn = LLTM_CUDA(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print(FORWARD_BACK_INFO.format(str(device), 'cuda', forward, backward))

"""
ssh://cc@129.114.108.89:22/home/cc/anaconda3/bin/python -u /home/cc/code/time-series-ml/cnns/nnlib/pytorch_cuda/benchmark.py
lltm cpu python: Forward: 44.325 sec | Backward 53.321 sec

"""
