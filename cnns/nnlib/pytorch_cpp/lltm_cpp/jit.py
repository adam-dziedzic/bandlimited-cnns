from torch.utils.cpp_extension import load

lltm_cpp = load(name="lltm_cpp", sources=["lltm.lltm_cpp"], verbose=True)
help(lltm_cpp)
