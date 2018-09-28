import torch
import time
from cnns.nnlib.pytorch_cuda.python.lltm import LLTM

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6 / 1e5,
                                                       backward * 1e6 / 1e5))

"""
Result python athena: Forward: 141.675 us | Backward 197.573 us

/home/adam/anaconda3/bin/python3.6 /home/adam/code/time-series-ml/cnns/nnlib/pytorch_cuda/python/benchmark.py
Forward: 147.558 us | Backward 200.779 us

Result cpp athena: /home/adam/anaconda3/bin/python3.6 /home/adam/code/time-series-ml/cnns/nnlib/pytorch_cuda/cpp/benchmark.py
Forward: 126.339 us | Backward 291.037 us

Result python chameleon: ssh://cc@129.114.108.89:22/home/cc/anaconda3/bin/python3.6 -u /home/cc/code/time-series-ml/cnns/nnlib/pytorch_cuda/python/benchmark.py
Forward: 276.649 us | Backward 438.827 us


"""
