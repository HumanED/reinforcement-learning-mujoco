# Training a robotic hand in simulation to rotate a cube to a target orientation (complete)

This project follows the paper [Learning Dexterity](https://arxiv.org/pdf/1808.00177) by OpenAI and is derived from the Gymnasium Robotics
[shadow dextrous hand](https://robotics.farama.org/envs/shadow_dexterous_hand/) environment with extensive modifications to follow the paper.

This project is complete for in-simulation only. We did not implement this on a real hand since we do not have access to computers powerful enough to train the AI for a physical hand. Therefore, we conclude this project and plan to move to the less computationally intensive [SO-100 arm](https://github.com/TheRobotStudio/SO-ARM100) project in 2025-2026 allowing for faster design iterations. 

### Installation (copied from Notion pages)
Install the miniconda package manager. https://anaconda.org/anaconda/conda.  
Open the Anaconda Prompt app or use the command line

Create a conda environment.
```
(base) C:\Users\ethan> conda create —-name HumanED python=3.10
C:\Users\ethan> conda activate HumanED
```

Move to the reinforcement-learning-mujoco directory and install requirements

`(HumanED) C:\Users\...\reinforcement-learning-mujoco> pip install -r requirements.txt`

Move to the Shadow_Gym2 directory and install the custom environment

`(HumanED) C:\Users\...\reinforcement-learning-mujoco\Shadow_Gym2> pip install -e .`

Install numpy 2 manually after `stable_baselines3` is installed. Make sure you are using numpy 2. There is a configuration problem where stable-baselines3 says it is incompatible with numpy 2 but it is actually fine in reality

`(HumanED) C:\Users..\> pip install numpy==2.2.1`

Run `python .\visualise_model.py` to check if everything works.  
Run `python .\evaluate_model.py` to compute model statistics like mean successes.\

Model statistics for the final model are
```
episode_rewards      mean: 170.446 std: 129.890 
success              mean: 23.180 std: 17.00
dropped              mean: 0.350 std: 0.48
dt                   mean: 0.080 std: 0.00 (ignore this figure it is just time between video frames)
total_timesteps      mean: 1043.130 std: 693.15
```