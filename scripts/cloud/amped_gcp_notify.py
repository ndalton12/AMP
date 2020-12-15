import os

import ray
from ray import tune
from ray.tune import TuneError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.algos.amped.amped_trainer import AmpedTrainer
from src.common.env_wrappers import register_super_mario_env


def train():
    register_super_mario_env()

    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

    ray.init(address="auto")

    def send_message(message):
        try:
            _ = client.chat_postMessage(channel='#notifications', text=message)
        except SlackApiError as e:
            print(f"Got an error: {e.response['error']}")

    try:
        tune.run(
            AmpedTrainer,
            config={
                "env": "super_mario",
                "framework": "torch",
                "num_workers": 16,
                "log_level": "INFO",
                "seed": 1337,
                # "num_envs_per_worker": 1,
                # "rollout_fragment_length": 256,
                "entropy_coeff": 0.01,
                "kl_coeff": 0.0,
                # "train_batch_size": 256,
                # "sgd_minibatch_size": 128,
                # "num_sgd_iter": 2,
                # "num_simulations": 25,
                "batch_mode": "complete_episodes",
                # #"remote_worker_envs": True,
                # "ignore_worker_failures": True,
                # #"num_cpus_per_worker": 1,
                "sample_async": True,
                #"no_done_at_end": True,
                # #"soft_horizon": True,
                #"horizon": 256,
                # "num_gpus": 1,
                # "mcts_param": {
                #     "k_sims": 5,
                # }
            },
            sync_config=tune.SyncConfig(upload_dir="gs://amp-results"),
            stop={"training_iteration": 500},
            checkpoint_freq=500,
            raise_on_failed_trial=True,
            checkpoint_at_end=True,
            #resume=True,
        )
    except TuneError as e:
        print(e)
        send_message("The trail failed :(")
    finally:
        send_message("Trial over")


if __name__ == "__main__":
    train()
