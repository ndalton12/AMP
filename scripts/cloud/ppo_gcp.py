
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from src.common.env_wrappers import register_super_mario_env


def train():
    register_super_mario_env()

    ray.init(address="auto")

    tune.run(
        PPOTrainer,
        config={
            "env": "super_mario",
            "framework": "torch",
            "num_workers": 4,
            "log_level": "INFO",
            "seed": 1337,
            "num_envs_per_worker": 5,
            "entropy_coeff": 0.01,
            "kl_coeff": 0.0,
            "num_sgd_iter": 2,
            "num_gpus": 1,
            "vf_share_layers": False,
        },
        sync_config=tune.SyncConfig(upload_dir="gs://amp-results"),
        stop={"training_iteration": 500},
        checkpoint_freq=500,
        checkpoint_at_end=True,
        #resume=True,
    )


if __name__ == "__main__":
    train()
