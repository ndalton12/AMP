import ray
from ray import tune

from src.algos.amped.amped_trainer import AmpedTrainer
from src.common.counter import Counter
from src.common.env_wrappers import register_super_mario_env


def train():
    register_super_mario_env()

    ray.init(address="auto")
    _ = Counter.options(name="global_counter", max_concurrency=1).remote()

    tune.run(
        AmpedTrainer,
        config={
            "env": "super_mario",
            "framework": "torch",
            "num_workers": 1,
            "log_level": "INFO",
            "seed": 1337,
            "num_envs_per_worker": 3,
            "entropy_coeff": 0.01,
            "kl_coeff": 0.0,
            "train_batch_size": 256,
            "num_sgd_iter": 2,
            "num_simulations": 10,
            "batch_mode": "truncate_episodes",
            "remote_worker_envs": True
            #"ignore_worker_failures": True,
        },
        stop={"episodes_total": 100},
        #checkpoint_freq=1,
        #checkpoint_at_end=True,
        #resume=True,
    )


if __name__ == "__main__":
    train()
