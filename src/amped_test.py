from ray import tune

from src.algos.amped.amped_trainer import AmpedTrainer
from src.common.env_wrappers import register_super_mario_env


def train():
    register_super_mario_env()

    tune.run(
        AmpedTrainer,
        config={
            "env": "super_mario",
            "framework": "torch",
            "num_workers": 1,
            "log_level": "DEBUG",
            "seed": 1337,
            "num_envs_per_worker": 5,
            "entropy_coeff": 0.01,
            "kl_coeff": 0.0,
            "train_batch_size": 256,
            "num_sgd_iter": 2,
            "num_simulations": 25,
            #"ignore_worker_failures": True,
        },
        stop={"episodes_total": 100},
        #checkpoint_freq=1,
        #checkpoint_at_end=True,
        #resume=True,
    )


if __name__ == "__main__":
    train()
