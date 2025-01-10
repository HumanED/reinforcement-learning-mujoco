import os
from typing import Optional
import mujoco
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from ..utils import rotations
from ..utils import mujoco_utils

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 55.0,
    "elevation": -25.0,
    "lookat": np.array([1, 0.96, 0.14]),
}

# Order of joints based on https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/
wrist_low = np.array([-0.489, -0.698], dtype=np.float32)
wrist_high = np.array([0.14, 0.489], dtype=np.float32)
first_low = np.array([-0.349, 0, 0], dtype=np.float32)
first_high = np.array([0.349, 1.571, 1.571], dtype=np.float32)
middle_low = np.array([-0.349, 0, 0], dtype=np.float32)
middle_high = np.array([0.349, 1.571, 1.571], dtype=np.float32)
ring_low = np.array([-0.349, 0, 0], dtype=np.float32)
ring_high = np.array([0.349, 1.571, 1.571], dtype=np.float32)
little_low = np.array([0, -0.349, 0, 0], dtype=np.float32)
little_high = np.array([0.785, 0.349, 1.571, 1.571], dtype=np.float32)
thumb_low = np.array([-1.047, 0, -0.209, -0.524, -1.571], dtype=np.float32)
thumb_high = np.array([1.047, 1.222, 0.209, 0.524, 0], dtype=np.float32)
action_low = np.concatenate([wrist_low, first_low, middle_low, ring_low, little_low, thumb_low])
action_high = np.concatenate([wrist_high, first_high, middle_high, ring_high, little_high, thumb_high])
number_of_bins = 11
bin_sizes = (action_high - action_low) / number_of_bins


class ShadowEnvMujoco(gymnasium.Env, EzPickle):
    """
    Gymnasium environment of the shadow dexterous hand based on implementation
    from https://github.com/Farama-Foundation/Gymnasium-Robotics
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 50,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
    ):
        """
        Initialize environment
        Args:
            render_mode (optional string): type of rendering mode, "human" for window rendering and "rgb_array" for offscreen. Defaults to None.
        """
        EzPickle.__init__(self, render_mode)

        # SETTINGS
        # Anything in all-caps must not be changed by the program itself during execution
        # self.step_limit (integer)         number of calls to step() before episode is cut short (truncates)
        # n_substeps (integer)              number of MuJoCo simulation timesteps per call to step()
        # step_between_goals                Robot has step_limit_value calls to step() to reach the current goal or the environment is truncated
        # screen_width                      screen_width of each rendered frame. Defaults to DEFAULT_SIZE.
        # screen_height                     screen_height of each rendered frame . Defaults to DEFAULT_SIZE.
        # rotation_threshold (float)        the threshold after which the rotation of a goal is considered achieved. Unit is radians
        # relative_control (bool)           Whether or not the hand is actuated in absolute joint positions or relative to the current joint position
        # randomize_initial_rotation (bool) Whether to set cube orientation to a random orientation at start of episode
        # fixed_goal                        Fixed Euler goal. Set to None if we are not using a random goal instead of a fixed goal
        # max_goals                         Maximum number of goals reached before truncating the environment
        # self.target_euler                 Fixed target orientation
        # n_actions (intege)                size of the action space.
        # n_obs (integer)                   number of observations
        # self.fullpath                     Path to Mujoco XML file holding all information on the simulation environment

        self.step_limit = np.inf
        self.STEP_BETWEEN_GOALS = 100 # 8 seconds real time
        self.N_SUBSTEPS = 40
        self.ROTATION_THRESHOLD = 0.4
        self.RELATIVE_CONTROL = False
        self.RANDOMIZE_INITIAL_ROTATION = True
        self.fixed_goal = None
        self.MAX_GOALS = 50


        n_actions = 20
        n_obs = 85
        self.action_space = gymnasium.spaces.MultiDiscrete(nvec=[11] * n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.fullpath = os.path.join(os.path.dirname(__file__), "../resources", "hand", "manipulate_block.xml")
        self.screen_width = 1200
        self.screen_height = 800

        # END SETTINGS
        self._load_mujoco_robot()
        self.step_between_goals_count = 0
        self.goal_counter = 0
        self.previous_angular_diff = np.pi
        self.goal = np.zeros(0)
        self.timesteps = 0
        self.info = {
            "success": 0,
            "dropped":False,
            "timesteps":0,
            "dt": 0
        }
        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            DEFAULT_CAMERA_CONFIG,
        )

    def _load_mujoco_robot(self):
        """
        Loads XML file containing all information about the cube and hand. Runs only once when gym.make() is called
        """
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        self._model_names = mujoco_utils.MujocoModelNames(self.model)
        self.fingertip_body_names = list(filter(lambda name: name.endswith("distal"), self._model_names.body_names))

        self.model.vis.global_.offwidth = self.screen_width
        self.model.vis.global_.offheight = self.screen_height

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
        Reset the environment
        """
        # Reset environment state
        super().reset(seed=seed)
        self.step_between_goals_count = 0
        self.goal_counter = 0
        self.timesteps = 0
        # Time between each frame in rendering
        dt = self.model.opt.timestep * self.N_SUBSTEPS
        self.info = {
            "success": 0,
            "dropped":False,
            "dt": dt,
            "timesteps":0
        }
        self.previous_angular_diff = np.pi
        self._reset_sim()

        # Compute initial goal
        self.goal = self._compute_goal()

        # Return obs and info
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self.info

    def _compute_goal(self):
        """Returns goal quaternion"""
        if self.fixed_goal:
            new_goal = self.fixed_goal.copy()
        else:
            # Generate a random goal
            new_goal = self.np_random.uniform(-np.pi, np.pi, size=3)
        target_quat = rotations.euler2quat(new_goal)
        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        return target_quat

    def _reset_sim(self):
        """Resets simulation and puts cube in a random orientation"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        initial_cube_qpos = mujoco_utils.get_joint_qpos(
            self.model, self.data, "object:joint"
        ).copy()
        initial_cube_pos, initial_cube_quat = initial_cube_qpos[:3], initial_cube_qpos[3:]
        assert initial_cube_qpos.shape == (7,)
        assert initial_cube_pos.shape == (3,)
        assert initial_cube_quat.shape == (4,)

        # Randomization initial rotation.
        if self.RANDOMIZE_INITIAL_ROTATION:
            # All possible rotations are allowed.
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            offset_quat = rotations.quat_from_angle_and_axis(angle, axis)
            initial_cube_quat = rotations.quat_mul(initial_cube_quat, offset_quat)

        initial_cube_quat /= np.linalg.norm(initial_cube_quat)
        initial_cube_qpos = np.concatenate([initial_cube_pos, initial_cube_quat])

        mujoco_utils.set_joint_qpos(self.model, self.data, "object:joint", initial_cube_qpos)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._apply_action(np.zeros(self.action_space.shape))
            try:
                mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)
            except Exception:
                return False

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (np.ndarray): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (float): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state (cube is dropped)
            truncated (boolean): Whether the agent exceeds maximum time for an episode
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        # self.timesteps += 1
        self.step_between_goals_count += 1
        self.info["timesteps"] += 1

        # Rescale the angle between -1 and 1. See action space of https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/
        # See second min-max normalization formula https://en.wikipedia.org/wiki/Feature_scaling
        action = -1 + (action * 2) / 10
        self._apply_action(action)

        obs = self._get_obs()

        terminated = truncated = False
        cube_quat_idx = 18
        current_angular_diff = rotations.angular_difference_abs(obs[cube_quat_idx: cube_quat_idx + 4], self.goal)
        if current_angular_diff < self.ROTATION_THRESHOLD:
            reward = 5
            self.info["success"] += 1
            # Select a new goal and reset the step_between_goals timer
            self.goal_counter += 1
            self.goal = self._compute_goal()
            self.step_between_goals_count = 0
            self.previous_angular_diff = np.pi # Without this, immediately when new goal is given a large negative reward appears
        else:
            reward = self.previous_angular_diff - current_angular_diff
            self.previous_angular_diff = current_angular_diff
        cube_posz_idx = 17
        if obs[cube_posz_idx] < -0.05:
            # Cube is dropped
            self.info["dropped"] = True
            terminated = True
            reward = -20
        # truncate environment when run out of time to reach current goal or hit maximum number of goals
        if self.step_between_goals_count > self.STEP_BETWEEN_GOALS or self.goal_counter >= self.MAX_GOALS:
            truncated = True
        # if  self.timesteps > self.step_limit:
        #     truncated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info

    def _apply_action(self, action):
        """Sends AI action to Mujoco simulation and steps simulation to execute the action"""
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0

        if self.RELATIVE_CONTROL:
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].replace(":A_", ":")
                )
            for joint_name in ["FF", "MF", "RF", "LF"]:
                act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
                actuation_center[act_idx] += self.data.get_joint_qpos(
                    f"robot0:{joint_name}J0"
                )
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)

    def _get_obs(self):
        """
        observation = [fingertip_pos, cube_qpos, self.goal, quat_diff, robot_qpos, robot_qvel]
        fingertip_pos:  15 numbers. Global position (x, y, z) of the 5 fingertips
        cube_qpos:      7 numbers. Position then by orientation quaternion. (x,y,z, qw, qx, qy, qz) where q- terms are quaternion.
        self.goal:      4 numbers. Goal rotation expressed as a quaternion (w, x, y, z)
        quat_diff:      4 numbers. Quaternion expressing relative orientation between orientation in cube_qpos and selfgoal. (w, x, y, z)
        robot_qpos:     24 numbers. Angles of each hinge joint in the hand in radians.
        robot_qvel:     24 numbers. Angular velocity of each hinge joint in the hand in radians per second.
        cube_qvel:      7 numbers. Linear and angular velocity of the cube. (dx, dy, dz, dqw, dqx, dqy, dqx)
        Indexing        [0 - 14 (fingertip_pos), 15 - 21 (cube_qpos), 22 - 25 (self.goal), 26 - 29 (quat_diff), 30 - 53 (robot_qpos), 54 - 77 (robot_qvel), 78 - 84 (cube_qvel)]
        """
        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        # Get position (x,y,z) of the 5 fingertips
        fingertip_pos = mujoco_utils.get_all_body_pos(self.model, self.data, self.fingertip_body_names).ravel()

        # cube_qvel_raw is linear velocity (x,y,z) then angular velocity (x, y,z). We need the angular velocity bit as a quaternion
        cube_qvel_raw = mujoco_utils.get_joint_qvel(self.model, self.data, "object:joint")
        cube_qvel = np.concatenate([cube_qvel_raw[:3], rotations.euler2quat_vel(cube_qvel_raw[3:])])

        # cube position and orientation. (x,y,z, qw, qx, qy, qz)
        cube_qpos = mujoco_utils.get_joint_qpos(self.model, self.data, "object:joint")

        quat_diff = rotations.quat_mul(cube_qpos[..., 3:], rotations.quat_conjugate(self.goal))
        observation = np.concatenate([fingertip_pos, cube_qpos, self.goal, quat_diff, robot_qpos, robot_qvel, cube_qvel])
        return observation

    # --- other utility methods
    def render(self):
        """Render a frame of the Mujoco simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.

        offset_pos = np.array([1, 0.87, 0.3])
        render_target = np.concatenate([offset_pos, self.goal])
        assert render_target.shape == (7,), f"Actual goal shape {render_target.shape}"

        mujoco_utils.set_joint_qpos(self.model, self.data, "target:joint", render_target)
        mujoco_utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

        if "object_hidden" in self._model_names.geom_names:
            hidden_id = self._model_names.geom_name2id["object_hidden"]
            self.model.geom_rgba[hidden_id, 3] = 1.0
        # TODO: Check if need to put N_SUBSTEPS in mj_forward
        mujoco.mj_forward(self.model, self.data)
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """
        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        Call this method to prevent errors when rendering.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
