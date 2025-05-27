# Training a robotic hand in simulation to rotate a cube to a target orientation (complete)

This project follows the paper [Learning Dexterity](https://arxiv.org/pdf/1808.00177) by OpenAI and is derived from the Gymnasium Robotics
[shadow dextrous hand](https://robotics.farama.org/envs/shadow_dexterous_hand/) environment with extensive modifications to follow the paper.

This project is complete for in-simulation only. We did not implement this on a real hand since we do not have access to computers powerful enough to train the AI for a physical hand.

### Installation (copied from Notion pages)
Create a conda environment

```
(base) C:\Users\ethan> conda create —-name HumanED python=3.10
C:\Users\ethan> conda activate HumanED
```

Move to the Shadow_gym2_project directory and install requirements

`(HumanED) C:\Users\...\Shadow_gym2_project> pip install -r requirements.txt`

Install numpy 2 manually after `stable_baselines3` is installed. Make sure you are using numpy 2. There is a configuration problem where stable-baselines3 says it is incompatible with numpy 2 but it is actually fine in reality

`(HumanED) C:\Users..\> pip install numpy==2.2.1`

Run `visualise_model.py` to check if everything works.
Run `evaluate_model.py` to compute model statistics like mean successes.