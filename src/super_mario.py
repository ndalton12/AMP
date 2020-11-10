from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env


def env_creator(env_config):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env


register_env("super_mario", env_creator)

tune.run(PPOTrainer, config={"env": "super_mario",
                             "framework": "torch",
                             "model": {
                                "dim": [240, 256],
                                "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1], [512, [50, 54], 1]],
                             }
                             })
