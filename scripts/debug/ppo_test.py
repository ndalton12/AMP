
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from src.common.env_wrappers import register_super_mario_env


def train():
    register_super_mario_env()

    tune.run(PPOTrainer, config={
        "env": "super_mario",
        "framework": "torch",
        "num_workers": 4,
        "log_level": "DEBUG",
        "seed": 1337,
        "num_envs_per_worker": 5,
        "entropy_coeff": 0.01,
        "vf_share_layers": True,
        "num_sgd_iter": 2,
    })


if __name__ == "__main__":
    train()
