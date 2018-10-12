import re
import sys
import doctest

log1 = """orch.utils.backcompat\n', 'import torch.onnx\n', 'import torch.jit\n', 'import torch.random\n', 'import torch.distributions\n', 'import torch.testing\n', 'import torch.backends.cuda\n', 'import torch.backends.mkl\n', 'from torch.autograd import no_grad, enable_grad, set_grad_enabled\n', '\n', '_C._init_names(list(torch._storage_classes))\n', '\n', '# attach docstrings to torch and tensor functions\n', 'from . import _torch_docs, _tensor_docs, _storage_docs\n', 'del _torch_docs, _tensor_docs, _storage_docs\n'], '/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/__init__.py'), 'get_objects.py': (1361, 1539294648.8218243, ['import gc\n', 'import torch\n', '\n', 'tensor1': tensor([1, 2])\n', 'tensor2 = torch.tensor([1, 2, 3, 4])\n', 'tensor3 = torch.tensor([1, 2, 3, 4, 5])\n', '\n', 'gc_objects = gc.get_objects()\n', 'print("length of gc objects: ", len(gc_objects))\n', 'obj1 = gc_objects[0]\n', 'print("obj1: ", obj1)\n', 'print("dir obj1: ", dir(obj1))\n', 'print("obj2: ", gc_objects[1])\n', '\n', 'with open("all_objects.txt", "w") as file:\n', '    for obj in gc.get_objects():\n', '        try:\n', '            file.write("obj: " + str(obj) + "\\n")\n', '            # if "tensor" in str(obj):\n', '            #     file.write("tensor in obj\\n")\n', '            # else:\n', '            #     file.write("no tensor in obj\\n")\n', '            # file.write("\\n\\n\\n\\n\\n")"""
log2 = """{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7ff0535ef9e8>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'get_objects.py', '__cached__': None, 'gc': <module 'gc' (built-in)>, 'torch': <module 'torch' from '/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/__init__.py'>, 'tensor1': tensor([1, 2]), 'tensor2': tensor([1, 2, 3, 4]), 'tensor3': tensor([1, 2, 3, 4, 5]), 'gc_objects': [['/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl'], <_frozen_importlib_external.SourceFileLoader object at 0x7ff03bd990f0>, ModuleSpec(name='torch.backends.mkl', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7ff03bd990f0>, origin='/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl/__init__.py', submodule_search_locations=['/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl']), {'name': 'torch.backends.mkl', 'loader': <_frozen_importlib_external.SourceFileLoader object at 0x7ff03bd990f0>, 'origin': '/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl/__init__.py', 'loader_state': None, 'submodule_search_locations': ['/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl'], '_set_fileattr': True, '_cached': '/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl/__pycache__/__init__.cpython-36.pyc', '_initializing': False}, <module 'torch.backends.mkl' from '/home/athena/adam/anaconda3/lib/python3.6/site-packages/torch/backends/mkl/__init__.py'>, {'__name__': 'torch.backends.mkl', '__doc__': None, '__package__': 'torch.backends.mkl', '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7ff03bd990f"""


def get_tensor_names(string_obj):
    """
    Get tensor names from the given string.

    :param obj: a string representing an object from gc.get_objects()
    :return: the name of the tensor variable

    >>> names = get_tensor_names(log2)
    >>> assert len(names) == 3
    >>> assert names[0] == "tensor1"
    >>> assert ['tensor1', 'tensor2', 'tensor3'] == names

    >>> names = get_tensor_names(log1)
    >>> assert len(names) == 1
    >>> assert ["tensor1"] == names

    """
    # There can be more than a single tensor name in the string.
    # This pattern does not overlap in the object from gc.
    pattern = "'(\w+)': tensor\("
    # pos = string_obj.find(pattern)
    return re.findall(pattern, string_obj)


def find_single_name(string_obj):
    # The name of the variable is before the first ' single quote.
    pattern = "': tensor("
    prev_pos = 0
    pos = string_obj.find(pattern)
    if pos != -1:
        var_name = []
        for char in reversed(string_obj[prev_pos:pos]):
            if char == "'":
                break
            else:
                var_name.insert(0, char)
        return "".join(var_name)
    return None


if __name__ == "__main__":
    print(get_tensor_names(log2))

    sys.exit(doctest.testmod()[0])
