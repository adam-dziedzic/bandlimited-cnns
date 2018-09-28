import torch
import time
from cnns.nnlib.pytorch_cuda.cpp.lltm import LLTM as LLTM_cpp
from cnns.nnlib.pytorch_cuda.python.lltm import LLTM as LLTM_py
from cnns.nnlib.pytorch_cuda.cuda.lltm import LLTM as LLTM_cuda

batch_size = 16
input_features = 32
state_size = 128

device = torch.device("cpu")

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)


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


rnn = LLTM_py(input_features, state_size, device=device)
forward, backward = test(X, h, C, device=device)
print('lltm cpu python: Forward: {:.3f} sec | Backward {:.3f} sec'.format(
    forward, backward))

rnn = LLTM_cpp(input_features, state_size, device=device)
forward, backward = test(X, h, C, device=device)
print('lltm cpu cpp: Forward: {:.3f} sec | Backward {:.3f} sec'.format(
    forward, backward))

device = torch.device("cuda")
X = X.to(device)
h = h.to(device)
C = C.to(device)

rnn = LLTM_py(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print('lltm gpu python: Forward: {:.3f} sec | Backward {:.3f} sec'.format(
    forward, backward))

rnn = LLTM_cpp(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print('lltm gpu cpp: Forward: {:.3f} sec | Backward {:.3f} sec'.format(
    forward, backward))

rnn = LLTM_cuda(input_features, state_size, device=device).to(device)
forward, backward = test(X, h, C, device=device)
print('lltm gpu cuda cpp: Forward: {:.3f} sec | Backward {:.3f} sec'.format(
    forward, backward))

"""
ssh://cc@129.114.108.89:22/home/cc/anaconda3/bin/python -u /home/cc/code/time-series-ml/cnns/nnlib/pytorch_cuda/benchmark.py
lltm cpu python: Forward: 44.325 sec | Backward 53.321 sec

"""
