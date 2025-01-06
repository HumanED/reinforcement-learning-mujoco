import mujoco
import mediapy as media
import subprocess
from utils import mujoco_utils
from utils.rotations import angular_difference_abs
import numpy as np
import cv2

model = mujoco.MjModel.from_xml_path("C:/Users/ethan/Documents/Edinburgh_Uni/HumanED/Shadow_gym2_project/Shadow-Gym2/envs/shadow_hand/resources/hand/manipulate_block.xml")
data = mujoco.MjData(model)
print(data.qpos, data.qpos.shape)
print(data.qvel, data.qvel.shape)
_model_names = mujoco_utils.MujocoModelNames(model)
# print(_model_names.body_names, len(_model_names.body_names))
print(_model_names.joint_names, len(_model_names.joint_names))

joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
joint_addr = model.jnt_qposadr[joint_id]
print(joint_addr)

# Numbers align with shadow_env.py's implementation
print(angular_difference_abs(np.array([0.1,0.1,0.2,0.3], dtype=np.float32), np.array([0.4,0.3,0.2,0.1],dtype=np.float32)))
print(angular_difference_abs(np.array([0.8,0.1,0.2,0.3],dtype=np.float32), np.array([0.4,0.3,0.2,0.1,],dtype=np.float32)))
print(angular_difference_abs(np.array([0.1,0.9,0.2,0.3],dtype=np.float32), np.array([0.4,0.3,0.2,0.1],dtype=np.float32)))




# print(_model_names.geom_names, len(_model_names.geom_names)) # geoms include items used for physics and items for visualisation. C and V items. geoms mainly for collision and visualisation
# cartesian position of body frame
# mujoco.mj_name2id
# link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, )
# joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
# data.xpos

# subprocess.run('C:/Users/ethan/Documents/Edinburgh_Uni/HumanED/mujoco-3.1.5-windows-x86_64/bin/simulate.exe "C:/Users/ethan/Documents/Edinburgh_Uni/HumanED/Shadow_gym2_project/Shadow-Gym2/envs/shadow_hand/resources/hand/manipulate_block.xml" ')
# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data)
#     image = renderer.render()
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # media.show_image(renderer.render()) # media is for Jupyter
#     cv2.imshow('Image', image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
