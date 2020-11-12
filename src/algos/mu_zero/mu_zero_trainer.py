from ray.rllib.agents.trainer_template import build_trainer

from src.algos.mu_zero.mu_config import DEFAULT_CONFIG
from src.algos.mu_zero.mu_zero_policy import MuZeroTorchPolicy
from src.algos.ppo.ppo_trainer import validate_config, execution_plan

MuZeroTrainer = build_trainer(
    name="MuZero",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=MuZeroTorchPolicy,
    execution_plan=execution_plan
)
