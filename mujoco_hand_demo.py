import time
import mujoco
import numpy as np
from Shadow_Gym2.shadow_gym.envs.shadow_env_mujoco import ShadowEnvMujoco

# Velocity limits
wrist_vel_low = np.array([-2.0, -2.0])
wrist_vel_high = np.array([2.0, 2.0])
index_vel_low = np.array([-2.0, -2.0, -2.0])
index_vel_high = np.array([2.0, 2.0, 2.0])
middle_vel_low = np.array([-2.0, -2.0, -2.0])
middle_vel_high = np.array([2.0, 2.0, 2.0])
ring_vel_low = np.array([-2.0, -2.0, -2.0])
ring_vel_high = np.array([2.0, 2.0, 2.0])
little_vel_low = np.array([-2.0, -2.0, -2.0, -2.0])
little_vel_high = np.array([2.0, 2.0, 2.0, 2.0])
thumb_vel_low = np.array([-4.0, -4.0, -4.0, -4.0, -4.0])
thumb_vel_high = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
hand_vel_low = np.concatenate((wrist_vel_low, index_vel_low, middle_vel_low, ring_vel_low, little_vel_low, thumb_vel_low))
hand_vel_high = np.concatenate((wrist_vel_high, index_vel_high, middle_vel_high, ring_vel_high, little_vel_high, thumb_vel_high))

# Order of joints based on https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/
wrist_low = np.array([-0.489, -0.785])
wrist_high = np.array([0.140, 0.524])
first_low = np.array([-0.349, 0.0, 0.0])
first_high = np.array([0.349, 1.571, 1.571])
middle_low = np.array([-0.349, 0.0, 0.0])
middle_high = np.array([0.349, 1.571, 1.571])
ring_low = np.array([-0.349, 0.0, 0.0])
ring_high = np.array([0.349, 1.571, 1.571])
little_low = np.array([0.0, -0.349, 0.0, 0.0])
little_high = np.array([0.785, 0.349, 1.571, 1.571])
thumb_low = np.array([-1.047, 0.0, -0.209, -0.524, -1.571])
thumb_high = np.array([1.047, 1.222, 0.209, 0.524, 0.0])
action_low = np.concatenate([wrist_low, first_low, middle_low, ring_low, little_low, thumb_low])
action_high = np.concatenate([wrist_high, first_high, middle_high, ring_high, little_high, thumb_high])
number_of_bins = 11
bin_sizes = (action_high - action_low) / number_of_bins
print('action shapes: ', action_high.shape, action_low.shape, bin_sizes.shape)

labels = ['wrist motion (horizontal)',
    'wrist motion (vertical)',
    'index finger (horizontal)',
    'index finger base (vertical)',
    'index finger middle (vertical)',
    'index finger tip (vertical)',
    'middle finger (horizontal)',
    'middle finger base (vertical)',
    'middle finger middle (vertical)',
    'middle finger tip (vertical)',
    'ring finger (horizontal)',
    'ring finger base (vertical)',
    'ring finger middle (vertical)',
    'ring finger tip (vertical)',
    'little finger grasp',
    'little finger (horizontal)',
    'little finger base (vertical)',
    'little finger middle (vertical)',
    'little finger tip (vertical)',
    'thumb rotation',
    'thumb finger base (vertical)',
    'thumb finger middle (vertical)',
    'thumb finger middle (horizontal)',
    'thumb finger tip (horizontal)',]

# Load env
test_env = ShadowEnvMujoco('human')
obs, info = test_env.reset() # best practice
time_between_frames = info["dt"]

def stress_test():
    n_joints = 20
    rest_pos = np.zeros(n_joints)
    MAX = 11
    MIN = -11

    def test_vel(x, joint):
        # x for either min or max action, joint is the the joint id
        print(f"joint {joint} ({labels[joint]})")

        # max or min action for ith joint
        action = np.zeros(n_joints)
        action[joint] = x
        print(f"full action: {action}")

        # apply action
        joint_obs, reward, terminated, truncated, info = test_env.step(action)

        # display velocity of joint
        joint_qvels = joint_obs[54:78]
        print(f"Velocity: {hand_vel_low[joint]:.3f} <= {joint_qvels[joint]} <= {hand_vel_high[joint]+0.001}")

        # check if in acceptable vel range
        if not (hand_vel_low[joint]-0.001 <= joint_qvels[joint] <= hand_vel_high[joint]+0.001):
            print(f"***VEL OUT OF BOUNDS***: {joint_qvels[joint]} ")

        print('-'*40)
        time.sleep(1)


    # for each joint
    for i in range(n_joints):
        # text max and min actions
        test_vel(MAX, i)
        test_env.step(rest_pos)
        test_vel(MIN, i)

    print("DONE")

stress_test()
test_env.close() # don't forget this too


# testing random stuff

# def test_action(joint):
#     # for i in range(4):
#     #     test_action = test_env.action_space.sample()
#     #     test_env.step(test_action)
#     #     print(f"action: {test_action}, {test_action[joint]}")
#     #     time.sleep(1)

#     test_action = np.zeros(20)
#     test_env.step(test_action)
#     time.sleep(1)

#     test_action[joint] = -11
#     #test_action = action_low + (bin_sizes / 2) + (bin_sizes * test_action)
#     test_env.step(test_action)
#     print(f"{labels[joint]}, action: {test_action}")
#     time.sleep(1)

#     test_action[joint] = 11
#     test_env.step(test_action)
#     print(f"{labels[joint]}, action: {test_action}")
#     time.sleep(1)

# for joint in range(20):
#     test_action(joint)