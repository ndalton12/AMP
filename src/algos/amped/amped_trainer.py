from ray.rllib.agents.trainer_template import build_trainer

from src.algos.amped.amped_policy import AMPED_CONFIG, AmpedTorchPolicy
from src.algos.ppo.ppo_trainer import validate_config, execution_plan

AmpedTrainer = build_trainer(
    name="Amped",
    default_config=AMPED_CONFIG,
    validate_config=validate_config,
    default_policy=AmpedTorchPolicy,
    execution_plan=execution_plan
)
