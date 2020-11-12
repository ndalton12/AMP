import functools
from typing import Optional, Dict, Type, Union, List

import gym

from ray.rllib import SampleBatch
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import setup_mixins, kl_and_loss_stats, vf_preds_fetches, KLCoeffMixin, \
    ValueNetworkMixin, training_view_requirements_fn
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy import build_torch_policy

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import sequence_mask

from src.algos.mu_zero.mu_config import DEFAULT_CONFIG
from src.algos.mu_zero.mu_mcts import MCTS
from ray.rllib.utils.typing import AgentID, TrainerConfigDict, TensorType

from src.algos.mu_zero.mu_model import make_mu_model

torch, _ = try_import_torch()


def postprocess_mu_zero(policy: Policy,
                        sample_batch: SampleBatch,
                        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
                        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:

    sample_batch = postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)

    return sample_batch


def setup_mixins_and_mcts(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:

    setup_mixins(policy, obs_space, action_space, config)

    policy.mcts = MCTS(policy.model, policy.config["mcts_param"], action_space.shape[0])


def training_view_requirements_mu_fn(policy: Policy) -> Dict[str, ViewRequirement]:
    reqs = training_view_requirements_fn(policy)
    reqs["mcts_policy"] = ViewRequirement(shift=0)

    return reqs


def fetch(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:

    """
    Stop trying to make fetch happen.
    """

    fetches = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)

    fetches["mcts_policy"] = input_dict["mcts_policy"]

    fetches[SampleBatch.ACTION_DIST_INPUTS] = input_dict[SampleBatch.ACTION_DIST_INPUTS]

    return fetches


def mu_zero_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:

    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    logp_ratio = torch.exp(logp - train_batch[SampleBatch.ACTION_LOGP])

    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    if policy.config["use_gae"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
        total_loss = reduce_mean_valid(
            -surrogate_loss + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy)
    else:
        mean_vf_loss = 0.0
        total_loss = reduce_mean_valid(-surrogate_loss +
                                       policy.kl_coeff * action_kl -
                                       policy.entropy_coeff * curr_entropy)

    pred_reward = model.reward_function()

    reward_loss = torch.nn.functional.mse_loss(pred_reward, train_batch[SampleBatch.REWARDS])

    mcts_loss = torch.nn.functional.nll(logp, train_batch["mcts_policy"])

    total_loss += reward_loss + mcts_loss

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl
    policy._mean_reward_loss = reward_loss
    policy._mcts_loss = mcts_loss

    return total_loss


def do_simulation(policy: Policy, model: ModelV2, input_dict, state_out):
    num_sims = policy.config["num_simulations"]

    for _ in range(num_sims):
        policy.mcts.simulation(input_dict[SampleBatch.CUR_OBS])

    dist_inputs = policy.mcts.get_root_policy(input_dict[SampleBatch.CUR_OBS])

    return dist_inputs, state_out


def mu_action_sampler(policy: Policy, model: ModelV2, input_dict, state_out, explore: bool, timestep):
    policy.exploration.before_compute_actions(explore=explore, timestep=timestep)

    dist_class = policy.dist_class
    dist_inputs, state_out = do_simulation(policy, model, input_dict, state_out)

    if not (isinstance(dist_class, functools.partial)
            or issubclass(dist_class, TorchDistributionWrapper)):
        raise ValueError(
            "`dist_class` ({}) not a TorchDistributionWrapper "
            "subclass! Make sure your `action_distribution_fn` or "
            "`make_model_and_action_dist` return a correct "
            "distribution class.".format(dist_class.__name__))

    action_dist = dist_class(dist_inputs, model)

    # Get the exploration action from the forward results.
    actions, logp = \
        policy.exploration.get_exploration_action(
            action_distribution=action_dist,
            timestep=timestep,
            explore=explore)

    input_dict[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs  # need this for PPO loss later on, so mutate here as
    # no real better spot without overwritting large parts of TorchPolicy

    return actions, logp, state_out


def mu_action_distribution(policy: Policy, model: ModelV2, current_obs, explore: bool, timestep, is_training):
    dist_class = policy.dist_class

    dist_inputs, state_out = model.get_policy(current_obs, explore)

    return dist_inputs, dist_class, state_out


MuZeroTorchPolicy = build_torch_policy(
    name="MuZeroTorchPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=mu_zero_loss,
    stats_fn=kl_and_loss_stats,
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
    make_model=make_mu_model,
)
