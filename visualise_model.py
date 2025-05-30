from stable_baselines3 import PPO
from Shadow_Gym2.shadow_gym.envs.shadow_env_mujoco import ShadowEnvMujoco
import numpy as np
import gymnasium
from gymnasium.wrappers import TransformObservation, NormalizeObservation
import os
import time

# SETTINGS
model_folder_zip = "PPO-33-shadowgym-ethan-FINAL/240000000FINAL.zip"


def make_env():
    """Creates gymnasium environment for visualisation with necessary wrappers"""

    def clip_observation(obs):
        """
        clips observation to within 5 standard deviations of the mean
        Refer to section D.1 of Open AI paper
        """
        return np.clip(obs, a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))

    # env = gymnasium.make("ShadowEnv-v1")
    env = ShadowEnvMujoco(render_mode="human")
    env = NormalizeObservation(env)
    env = TransformObservation(env, clip_observation, env.observation_space)
    return env


def main():
    model_path = os.path.join(os.path.dirname(__file__), "models", model_folder_zip)
    if not os.path.exists(model_path):
        raise Exception("Error: model not found")
    env = make_env()
    model = PPO.load(model_path, env=env)


    while True:
        terminated = False
        truncated = False
        episode_reward = 0
        obs, info = env.reset()
        time_between_frames = info["dt"]
        # Each frame should have a gap of 80ms for the visualisation video to match real time. Each frame represents simulation moving by 80ms
        # The time.sleep delay ensures the simulation moves at same speed as if it were a real robot. info["dt"] should be 0.08
        previous_success = 0

        print("DEBUG INFO")
        # print(f"Target rotation {info['goal_rotation']}")
        while not terminated and not truncated:
            start_time = time.time()
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            time_to_process = time.time() - start_time
            episode_reward += reward
            delay_time = time_between_frames - time_to_process
            if (delay_time > 0):
                time.sleep(delay_time)  # proper time
            if info["success"] > previous_success:
                print(f"Target rotation reached. Total success {info['success']}")
                previous_success = info["success"]
                time.sleep(0.5) # Pause a bit before moving to new goal

        print(f"Episode complete. episode_reward:{episode_reward:.3f}| success {info['success']} | dropped {info['dropped']} |")
        print("")
        time.sleep(1.5)  # Pause a bit before resetting environment


if __name__ == "__main__":
    main()
