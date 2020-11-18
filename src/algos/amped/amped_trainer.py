from ray.rllib.agents.ppo.ppo import validate_config
from ray.rllib.agents.trainer_template import build_trainer

from src.algos.amped.amped_policy import AMPED_CONFIG, AmpedTorchPolicy
from src.algos.mu_zero.mu_zero_trainer import mu_zero_execution_plan


AmpedTrainer = build_trainer(
    name="Amped",
    default_config=AMPED_CONFIG,
    validate_config=validate_config,
    default_policy=AmpedTorchPolicy,
    execution_plan=mu_zero_execution_plan
)
