from Shadow_Gym2.envs.shadow_hand.mujoco_backend import MujocoRobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.normalize import NormalizeObservation
import numpy as np

import os
# SETTINGS
# RecurrentPPO or PPO
recurrent = False
vectorized_env = True  # Set to True to use multiple environments
normalized_env = False
start_from_existing = False
existing_model_file = "11 500 000"  # no need .zip extension
# When you want to train PPO-20-shadowgym-ethan more and create PPO-21-shadowgym-ethan
# Set new_run_name="PPO-21-shadowgym-ethan" and run_name="PPO-20-shadowgym-ethan"
# A PPO-21 folder is created for the logs and models of the rerun.
new_run_name = "PPO-21-shadowgym-ethan"

# Run name should have model, unique number, and optionally a description
run_name = "PPO" + "-" + "0" + "-" + "shadowgym2-ethan"
saving_timesteps_interval = 25_000
start_saving = 200_000

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
normalize_stats = os.path.join(os.path.dirname(__file__), 'normalize_stats')
if not start_from_existing and os.path.exists(f"{models_dir}/{run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if not start_from_existing and os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")
if not start_from_existing and os.path.exists(f"{normalize_stats}/{run_name}"):
    raise Exception("Error: normalize_stats folder already exists. Change run_name")
if normalized_env:
    os.makedirs(f"{normalize_stats}/{run_name}")

class ConsoleLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True
    def __init__(self, verbose=0):
        super(ConsoleLoggerCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> None:
        # Log relevant metrics to console after each rollout
        print(f"Timesteps: {self.num_timesteps}")
        print(f"Episode Length Mean: {self.locals.get('ep_len_mean', None)}")
        print(f"Episode Reward Mean: {self.locals.get('ep_rew_mean', None)}")
        print(f"Approx KL: {self.logger.get('train/approx_kl', None)}")
        print(f"Clip Fraction: {self.logger.get('train/clip_fraction', None)}")
        print(f"Entropy Loss: {self.logger.get('train/entropy_loss', None)}")
        print(f"Explained Variance: {self.logger.get('train/explained_variance', None)}")
        print(f"Learning Rate: {self.logger.get('train/learning_rate', None)}")
        print(f"Loss: {self.logger.get('train/loss', None)}")
        print(f"Policy Gradient Loss: {self.logger.get('train/policy_gradient_loss', None)}")
        print(f"Value Loss: {self.logger.get('train/value_loss', None)}")

    def _on_training_end(self) -> None:
        # Log final metrics to console at the end of training
        print("Training ended.")
        print(f"Total Timesteps: {self.num_timesteps}")
def clip_observation(obs):
    """
    clips observation to within 5 standard deviations of the mean
    Refer to section D.1 of Open AI paper
    """
    return np.clip(obs,a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))
if __name__ == "__main__":
    rewards_callback = None
    num_envs = 20  # Number of parallel environments

    if vectorized_env:
        def make_env():
            env = MujocoRobot(render_mode=None)
            env = NormalizeObservation(env)
            env = TransformObservation(env,f=clip_observation)
            env = Monitor(env)
            return env
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        if normalized_env:
            env = VecNormalize(env)
        rewards_callback = ConsoleLoggerCallback()
    else:
        env = MujocoRobot(render_mode=None)
        env = Monitor(env)

    full_model_path = None
    if start_from_existing:
        full_model_path = os.path.join(models_dir, run_name, existing_model_file)
    if recurrent:
        from sb3_contrib import RecurrentPPO
        if start_from_existing:
            model = RecurrentPPO.load(full_model_path, env)
        else:
            model = RecurrentPPO(policy="MlpLstmPolicy", env=env, tensorboard_log=logs_dir, verbose=1)
    else:
        if start_from_existing:
            model = PPO.load(full_model_path, env)
        else:
            model = PPO(policy="MlpPolicy", env=env, tensorboard_log=logs_dir, verbose=1)

    console_logger = ConsoleLoggerCallback()

    timesteps = 0
    while True:
        model.learn(saving_timesteps_interval, tb_log_name=new_run_name, reset_num_timesteps=False) #, callback=console_logger)
        timesteps += saving_timesteps_interval
        if timesteps >= start_saving:
            model.save(f"{models_dir}/{new_run_name}/{timesteps}")
            if vectorized_env and normalized_env:
                normalize_stats_path = os.path.join(normalize_stats, run_name, str(timesteps) + '.pkl')
                env.save(normalize_stats_path)
