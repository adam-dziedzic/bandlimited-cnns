import gc
import torch

tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([1, 2, 3, 4])
tensor3 = torch.tensor([1, 2, 3, 4, 5])

gc_objects = gc.get_objects()
print("length of gc objects: ", len(gc_objects))
obj1 = gc_objects[0]
print("obj1: ", obj1)
print("dir obj1: ", dir(obj1))
print("obj2: ", gc_objects[1])

with open("all_objects.txt", "w") as file:
    for obj in gc.get_objects():
        try:
            file.write("obj: " + str(obj) + "\n")
            # if "tensor" in str(obj):
            #     file.write("tensor in obj\n")
            # else:
            #     file.write("no tensor in obj\n")
            # file.write("\n\n\n\n\n")
        except NameError:
            pass

for obj in gc.get_objects():
    try:
        if "': tensor(" in str(obj):
            with open("result_tensor6.txt", "a") as file:
                file.write(str(obj) + "\n")
            print(obj)
    except NameError:
        pass

# with open("all_objects.txt") as f:
#     for line in f:
#         if "tensor" in line:
#             print(line)

# for obj in gc.get_objects():
#     if "__str__" in dir(obj):
#         print("dir obj: ", dir(obj))
#         try:
#             print("obj: ", str(obj))
#         except NameError:
#             pass

#     if hasattr(obj, '__str__'):
#         print(obj)
    # if "tensor" in str(obj):
        # dir(obj)
        # print(str(obj))
