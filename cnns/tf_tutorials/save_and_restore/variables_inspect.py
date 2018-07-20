# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

print("print all tensors in checkpoint file")
chkp.print_tensors_in_checkpoint_file("./variables_model.ckpt", tensor_name="",
                                      all_tensors=True)

print("print only tensor v1 in checkpoint file")
chkp.print_tensors_in_checkpoint_file("./variables_model.ckpt",
                                      tensor_name="v1", all_tensors=False)

print("print only tensor v2 in checkpoint file")
chkp.print_tensors_in_checkpoint_file("./variables_model.ckpt",
                                      tensor_name="v2", all_tensors=False)


