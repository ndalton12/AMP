from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy import build_torch_policy
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule

from src.algos.amped.nomad_model import make_nomad_model
from src.algos.mu_zero.mu_config import DEFAULT_CONFIG
from src.algos.mu_zero.mu_zero_policy import mu_zero_loss, stats_function, fetch, postprocess_mu_zero, \
    setup_mixins_and_mcts, training_view_requirements_mu_fn, mu_action_sampler, mu_action_distribution

AMPED_CONFIG = DEFAULT_CONFIG
AMPED_CONFIG["mcts_param"]["order"] = 3

AmpedTorchPolicy = build_torch_policy(
    name="AmpedTorchPolicy",
    get_default_config=lambda: AMPED_CONFIG,
    loss_fn=mu_zero_loss,
    stats_fn=stats_function,
    extra_action_out_fn=fetch,
    postprocess_fn=postprocess_mu_zero,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins_and_mcts,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
    training_view_requirements_fn=training_view_requirements_mu_fn,
    action_sampler_fn=mu_action_sampler,
    action_distribution_fn=mu_action_distribution,
    make_model=make_nomad_model,
)