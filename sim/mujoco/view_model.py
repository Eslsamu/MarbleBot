import mujoco_py as mjc
import sys
import numpy as np

print(sys.argv[1])
model = mjc.load_model_from_path(sys.argv[1])
sim = mjc.MjSim(model)
act_shape = sim.data.ctrl.shape
state_shape = np.concatenate([
       sim.data.qpos.flat,
       sim.data.qvel.flat]).shape
viewer = mjc.MjViewer(sim)
