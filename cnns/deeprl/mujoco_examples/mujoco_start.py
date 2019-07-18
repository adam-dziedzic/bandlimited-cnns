import mujoco_py
from os.path import dirname
import os

print("path: ", os.environ["PATH"])

model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

sim.step()
print(sim.data.qpos)
# [  2.09217903e-06  -1.82329050e-12  -1.16711384e-07  -4.69613872e-11
#   -1.43931860e-05   4.73350204e-10  -3.23749942e-05  -1.19854057e-13
#   -2.39251380e-08  -4.46750545e-07   1.78771599e-09  -1.04232280e-08]