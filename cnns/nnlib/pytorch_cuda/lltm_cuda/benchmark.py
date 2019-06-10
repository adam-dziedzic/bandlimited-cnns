import torch
import time
from cnns.nnlib.pytorch_cuda.lltm_cuda.lltm import LLTM

batch_size = 16
input_features = 32
state_size = 128

if torch.cuda.is_available():
    print("Cuda is available.")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

X = torch.randn(batch_size, input_features, device=device)
h = torch.randn(batch_size, state_size, device=device)
C = torch.randn(batch_size, state_size, device=device)

rnn = LLTM(input_features, state_size, device=device)

forward = 0
backward = 0
for _ in range(100000):  # 100000
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6 / 1e5,
                                                       backward * 1e6 / 1e5))

"""
100000 repetitions cuda
skr-compute1
ssh://ady@skr-compute1.cs.uchicago.edu:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/code/bandlimited-cnns-pycharm-win/cnns/nnlib/pytorch_cuda/lltm_cuda/benchmark.py
Cuda is available.
Forward: 281.410 us | Backward 690.074 us

100000 repetitions pytorch cuda
Result python skr-compute1: 
ssh://ady@skr-compute1.cs.uchicago.edu:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/code/bandlimited-cnns-pycharm-win/cnns/nnlib/pytorch_python/benchmark.py
Cuda is available.
Forward: 465.526 us | Backward 846.361 us

Result python athena: Forward: 141.675 us | Backward 197.573 us

Result lltm_cpp athena: Forward: 132.657 us | Backward 304.706 us

Result lltm_cpp athena: /home/adam/anaconda3/bin/python3.6 /home/adam/code/time-series-ml/cnns/nnlib/pytorch_cuda/lltm_cpp/benchmark_small.py
Forward: 126.339 us | Backward 291.037 us

1000 repetitions
ssh://ady@skr-compute1.cs.uchicago.edu:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/code/bandlimited-cnns-pycharm-win/cnns/nnlib/pytorch_cuda/lltm_cuda/benchmark.py
Cuda is available.
Forward: 6.406 us | Backward 7.766 us

"""
