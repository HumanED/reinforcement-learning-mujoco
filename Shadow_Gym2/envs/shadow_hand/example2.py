from mujoco_backend import MujocoRobot

env = MujocoRobot(render_mode="human")
print("observation space", env.observation_space.shape)
print("action space", env.action_space.shape)
obs = env.reset()

for i in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("obs", obs)
    print("rew", reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("info", info)
    print("###")
    if terminated or truncated:
        obs = env.reset()
        print("RESET")

env.close()
print("END")