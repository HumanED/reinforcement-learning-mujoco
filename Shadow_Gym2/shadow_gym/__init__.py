import gymnasium
from gymnasium.envs.registration import register
register(
    id='ShadowEnv-v1',
    entry_point='shadow_gym.envs:ShadowEnvMujoco'
)
