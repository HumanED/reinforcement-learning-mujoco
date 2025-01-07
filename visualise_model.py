from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Shadow_Gym2.shadow_gym.envs.shadow_env_mujoco import ShadowEnvMujoco
import numpy as np
import gymnasium
from gymnasium.wrappers import TransformObservation, NormalizeObservation
import os
import time

# SETTINGS
recurrent = False
vectorized_env = False
normalized_env = False
# Set num_evaluate to -1 to enable rendering and just view the project
num_evaluate = -1
# Run name should have model, unique number, and optionally a description
run_name = "PPO-24-shadowgym-ethan"
model_file = "5200000.zip"
normalize_stats_file = "1800000.pkl"


# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__),'models')
logs_dir = os.path.join(os.path.dirname(__file__),'logs')
normalize_stats = os.path.join(os.path.dirname(__file__), 'normalize_stats',run_name,normalize_stats_file)
model_path = f"{models_dir}/{run_name}/{model_file}"
if not os.path.exists(model_path):
    raise Exception("Error: model not found")

def clip_observation(obs):
    """
    clips observation to within 5 standard deviations of the mean
    Refer to section D.1 of Open AI paper
    """
    return np.clip(obs,a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))


GUI=False
if num_evaluate == -1:
    GUI = True

if vectorized_env:
    env = DummyVecEnv([lambda : gymnasium.make("ShadowEnv-v0",GUI=GUI)])
    if normalized_env:
        env = VecNormalize.load(normalize_stats, env)
        env.training = False
        env.norm_reward = False
else:
    env = ShadowEnvMujoco(render_mode="human")
    env = NormalizeObservation(env)
    env = TransformObservation(env, clip_observation, env.observation_space)

if recurrent:
    from sb3_contrib import RecurrentPPO
    print("Rendering recurrentPPO model...")
    model = RecurrentPPO.load(model_path, env=env)
    num_envs = 1
    lstm_states = None
    episode_starts = np.ones((num_envs,),dtype=bool)
    obs, _ = env.reset()
    while True:
        action, lstm_states = model.predict(obs,state=lstm_states,episode_start=episode_starts, deterministic=True)
        obs, rewards, episode_starts, info = env.step(action)
        time.sleep(1/60)

else:
    print("Running non recurrent PPO model")
    model = PPO.load(model_path,env=env)

    total_success = 0
    episode_count = 0
    if num_evaluate == -1:
        run_forever=True
    else:
        run_forever=False
    while episode_count < num_evaluate or run_forever:
        terminated = False
        truncated = False
        episode_reward = 0
        obs, _ = env.reset()
        while not terminated and not truncated:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            time.sleep(0.03)# proper time
        print(f"episode_reward:{episode_reward:.3f} info {info}")
        time.sleep(1) # Pause a bit before resetting environment
        if vectorized_env:
            # In vectorized environments, a list of infos is returned. We only want the first info.
            info = info[0]
        if info["success"]:
            total_success += 1
        episode_count += 1
        print(f"total_success:{total_success} episodes:{episode_count} ratio:{total_success / episode_count}")




