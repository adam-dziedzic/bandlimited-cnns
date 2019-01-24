import numpy as np
import torch
import torch_dct as dct

x = torch.randn(200)
X = dct.dct(x)   # DCT-II done through the last dimension
y = dct.idct(X)  # scaled DCT-III done through the last dimension
print("numerical error: ", (torch.abs(x - y)).sum())
assert (torch.abs(x - y)).sum() < 1e-4  # x == y within numerical tolerance

