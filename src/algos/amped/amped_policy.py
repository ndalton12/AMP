import gym
from ray.rllib import Policy
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin, ValueNetworkMixin, setup_mixins
from ray.rllib.policy import build_torch_policy
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils.typing import TrainerConfigDict

from src.algos.amped.nomad_mcts import NomadMCTS
from src.algos.amped.nomad_model import make_nomad_model
from src.algos.mu_zero.mu_config import DEFAULT_CONFIG
from src.algos.mu_zero.mu_zero_policy import mu_zero_loss, stats_function, fetch, postprocess_mu_zero, \
    mu_action_sampler, mu_action_distribution

AMPED_CONFIG = DEFAULT_CONFIG
AMPED_CONFIG["mcts_param"]["order"] = 3
AMPED_CONFIG["use_tpu"] = False
AMPED_CONFIG["surrogate_coeff"] = 1.0


def setup_nomad(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:

    setup_mixins(policy, obs_space, action_space, config)

    # assumed discrete action space
    policy.mcts = NomadMCTS(policy.model, policy.config["mcts_param"], action_space.n, policy.device)


AmpedTorchPolicy = build_torch_policy(
    name="AmpedTorchPolicy",
    get_default_config=lambda: AMPED_CONFIG,
    loss_fn=mu_zero_loss,
    stats_fn=stats_function,
    extra_action_out_fn=fetch,
    postprocess_fn=postprocess_mu_zero,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_nomad,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
    action_sampler_fn=mu_action_sampler,
    action_distribution_fn=mu_action_distribution,
    make_model=make_nomad_model,
)
