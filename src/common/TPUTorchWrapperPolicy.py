import time
from typing import Dict, Callable, Type, Union, List, Optional, Tuple

import gym
import ray
import torch
from ray.rllib import TorchPolicy, Policy, SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

import torch_xla.core.xla_model as xm


class TPUTorchWrapperPolicy(TorchPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: TrainerConfigDict,
            *,
            model: ModelV2,
            loss: Callable[[
                Policy, ModelV2, Type[TorchDistributionWrapper], SampleBatch
            ], Union[TensorType, List[TensorType]]],
            action_distribution_class: Type[TorchDistributionWrapper],
            action_sampler_fn: Optional[Callable[[
                TensorType, List[TensorType]
            ], Tuple[TensorType, TensorType]]] = None,
            action_distribution_fn: Optional[Callable[[
                Policy, ModelV2, TensorType, TensorType, TensorType
            ], Tuple[TensorType, Type[TorchDistributionWrapper], List[
                TensorType]]]] = None,
            max_seq_len: int = 20,
            get_batch_divisibility_req: Optional[Callable[[Policy],
                                                          int]] = None,
    ):
        """Build a policy from policy and loss torch modules.

        Note that model will be placed on GPU device if CUDA_VISIBLE_DEVICES
        is set. Only single GPU is supported for now.

        Args:
            observation_space (gym.spaces.Space): observation space of the
                policy.
            action_space (gym.spaces.Space): action space of the policy.
            config (TrainerConfigDict): The Policy config dict.
            model (ModelV2): PyTorch policy module. Given observations as
                input, this module must return a list of outputs where the
                first item is action logits, and the rest can be any value.
            loss (Callable[[Policy, ModelV2, Type[TorchDistributionWrapper],
                SampleBatch], Union[TensorType, List[TensorType]]]): Callable
                that returns a single scalar loss or a list of loss terms.
            action_distribution_class (Type[TorchDistributionWrapper]): Class
                for a torch action distribution.
            action_sampler_fn (Callable[[TensorType, List[TensorType]],
                Tuple[TensorType, TensorType]]): A callable returning a
                sampled action and its log-likelihood given Policy, ModelV2,
                input_dict, explore, timestep, and is_training.
            action_distribution_fn (Optional[Callable[[Policy, ModelV2,
                Dict[str, TensorType], TensorType, TensorType],
                Tuple[TensorType, type, List[TensorType]]]]): A callable
                returning distribution inputs (parameters), a dist-class to
                generate an action distribution object from, and
                internal-state outputs (or an empty list if not applicable).
                Note: No Exploration hooks have to be called from within
                `action_distribution_fn`. It's should only perform a simple
                forward pass through some model.
                If None, pass inputs through `self.model()` to get distribution
                inputs.
                The callable takes as inputs: Policy, ModelV2, input_dict,
                explore, timestep, is_training.
            max_seq_len (int): Max sequence length for LSTM training.
            get_batch_divisibility_req (Optional[Callable[[Policy], int]]]):
                Optional callable that returns the divisibility requirement
                for sample batches given the Policy.
        """
        self.framework = "torch"
        Policy.__init__(self, observation_space, action_space, config)

        counter = ray.get_actor("global_counter")
        ray.get(counter.inc.remote(1))
        count = ray.get(counter.get.remote())
        print(f"{count}********************")
        self.device = xm.xla_device(n=count)  # DIFFERENCE HERE FOR TPU USAGE

        self.model = model.to(self.device)
        # Combine view_requirements for Model and Policy.
        self.training_view_requirements = dict(
            **{
                SampleBatch.ACTIONS: ViewRequirement(
                    space=self.action_space, shift=0),
                SampleBatch.REWARDS: ViewRequirement(shift=0),
                SampleBatch.DONES: ViewRequirement(shift=0),
            },
            **self.model.inference_view_requirements)

        self.exploration = self._create_exploration()
        self.unwrapped_model = model  # used to support DistributedDataParallel
        self._loss = loss
        self._optimizers = force_list(self.optimizer())

        self.dist_class = action_distribution_class
        self.action_sampler_fn = action_sampler_fn
        self.action_distribution_fn = action_distribution_fn

        # If set, means we are using distributed allreduce during learning.
        self.distributed_world_size = None

        self.max_seq_len = max_seq_len
        self.batch_divisibility_req = get_batch_divisibility_req(self) if \
            callable(get_batch_divisibility_req) else \
            (get_batch_divisibility_req or 1)

    @override(Policy)
    def learn_on_batch(
            self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        # Get batch ready for RNNs, if applicable.
        pad_batch_to_sequences_of_same_size(
            postprocessed_batch,
            max_seq_len=self.max_seq_len,
            shuffle=False,
            batch_divisibility_req=self.batch_divisibility_req,
            _use_trajectory_view_api=self.config["_use_trajectory_view_api"],
        )

        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        # Calculate the actual policy loss.
        loss_out = force_list(
            self._loss(self, self.model, self.dist_class, train_batch))

        # Call Model's custom-loss with Policy loss outputs and train_batch.
        if self.model:
            loss_out = self.model.custom_loss(loss_out, train_batch)

        # Give Exploration component that chance to modify the loss (or add
        # its own terms).
        if hasattr(self, "exploration"):
            loss_out = self.exploration.get_exploration_loss(
                loss_out, train_batch)

        assert len(loss_out) == len(self._optimizers)

        # assert not any(torch.isnan(l) for l in loss_out)
        fetches = self.extra_compute_grad_fetches()

        # Loop through all optimizers.
        grad_info = {"allreduce_latency": 0.0}

        for i, opt in enumerate(self._optimizers):
            # Erase gradients in all vars of this optimizer.
            opt.zero_grad()
            # Recompute gradients of loss over all variables.
            loss_out[i].backward(retain_graph=(i < len(self._optimizers) - 1))
            grad_info.update(self.extra_grad_process(opt, loss_out[i]))

            if self.distributed_world_size:
                grads = []
                for param_group in opt.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            grads.append(p.grad)

                start = time.time()
                if torch.cuda.is_available():
                    # Sadly, allreduce_coalesced does not work with CUDA yet.
                    for g in grads:
                        torch.distributed.all_reduce(
                            g, op=torch.distributed.ReduceOp.SUM)
                else:
                    torch.distributed.all_reduce_coalesced(
                        grads, op=torch.distributed.ReduceOp.SUM)

                for param_group in opt.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            p.grad /= self.distributed_world_size

                grad_info["allreduce_latency"] += time.time() - start

        # Step the optimizer
        for i, opt in enumerate(self._optimizers):
            xm.optimizer_step(opt, barrier=True)  # HERE IS THE DIFFERENCE FOR TPU USE

        grad_info["allreduce_latency"] /= len(self._optimizers)
        grad_info.update(self.extra_grad_info(train_batch))
        if self.model:
            grad_info["model"] = self.model.metrics()
        return dict(fetches, **{LEARNER_STATS_KEY: grad_info})
