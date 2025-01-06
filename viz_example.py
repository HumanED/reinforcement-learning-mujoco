from stable_baselines3 import PPO
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.normalize import NormalizeObservation
from Shadow_Gym2.envs.shadow_hand.mujoco_backend import MujocoRobot
import numpy as np
import time
def clip_observation(obs):
    """
    clips observation to within 5 standard deviations of the mean
    Refer to section D.1 of Open AI paper
    """
    return np.clip(obs,a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))

model = PPO.load("models/PPO-23-shadowgym-ethan/10875000.zip")
env = MujocoRobot(render_mode="human")
env = NormalizeObservation(env)
env = TransformObservation(env, f=clip_observation)
for i in range(1_000_000):
    obs, _ = env.reset()
    terminated = truncated = False
    episode_reward = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        time.sleep(0.01)
    time.sleep(1)
    print(f"episode reward {episode_reward}")

env.close()