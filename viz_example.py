from stable_baselines3 import PPO
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers import NormalizeObservation
from Shadow_Gym2.shadow_gym.envs.shadow_env_mujoco import ShadowEnvMujoco
import numpy as np
import time
def clip_observation(obs):
    """
    clips observation to within 5 standard deviations of the mean
    Refer to section D.1 of Open AI paper
    """
    return np.clip(obs,a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))

model = PPO.load("models/PPO-23-shadowgym-ethan/10875000.zip")
env = ShadowEnvMujoco(render_mode="human")
env = NormalizeObservation(env)
env = TransformObservation(env, clip_observation, env.observation_space)
for i in range(1_000_000):
    obs, info = env.reset()
    terminated = truncated = False
    episode_reward = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        time.sleep(0.03)
    time.sleep(1)
    print(f"episode reward {episode_reward} info {info}")

env.close()