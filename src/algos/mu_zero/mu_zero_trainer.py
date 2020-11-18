
from ray.rllib.agents.ppo.ppo import UpdateKL, warn_about_bad_reward_scales, validate_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, SelectExperiences, ConcatBatches, StandardizeFields
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator

from src.algos.mu_zero.mu_config import DEFAULT_CONFIG
from src.algos.mu_zero.mu_zero_policy import MuZeroTorchPolicy


def mu_zero_execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the PPO algorithm. Defines the distributed dataflow.

    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    # Concatenate the SampleBatches into one.
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    # Standardize advantages.
    rollouts = rollouts.for_each(StandardizeFields(["advantages"]))
    # Standardize value targets as well
    rollouts = rollouts.for_each(StandardizeFields(["value_targets"]))

    # Perform one training step on the combined + standardized batch.
    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"],
                framework=config.get("framework")))

    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    # Warn about bad reward scales and return training metrics.
    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))


MuZeroTrainer = build_trainer(
    name="MuZero",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=MuZeroTorchPolicy,
    execution_plan=mu_zero_execution_plan
)
