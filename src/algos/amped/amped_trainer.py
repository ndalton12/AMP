from ray.rllib.agents.ppo.ppo import validate_config
from ray.rllib.agents.trainer_template import build_trainer

from src.algos.amped.amped_policy import AMPED_CONFIG, AmpedTorchPolicy
from src.algos.mu_zero.mu_zero_trainer import mu_zero_execution_plan


def TPUMixinAmped(config):
    if "use_tpu" in config and config["use_tpu"]:
        from src.algos.amped.amped_policy import get_amped_policy_tpu
        return get_amped_policy_tpu()
    else:
        return None


AmpedTrainer = build_trainer(
    name="Amped",
    default_config=AMPED_CONFIG,
    validate_config=validate_config,
    get_policy_class=TPUMixinAmped,
    default_policy=AmpedTorchPolicy,
    execution_plan=mu_zero_execution_plan
)
