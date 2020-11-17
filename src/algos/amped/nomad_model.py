from typing import Dict, List

import gym
import torch
from ray.rllib import SampleBatch
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.torch.misc import same_padding, SlimConv2d, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

import numpy as np

from src.algos.mu_zero.mu_model import MuZeroModel, MuZeroPredictionModel, MuZeroDynamicsModel


class NomadDynamicsModel(nn.Module):
    pass


class NomadModel(MuZeroModel):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, base_model: ModelV2, order: int):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.activation = base_model.model_config.get("conv_activation")
        self.output_size = base_model.num_outputs
        self.base_model = base_model
        self.order = order

        filters = self.model_config["conv_filters"]
        out_channels, kernel, stride = filters[-1]
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]

        self.prediction = MuZeroPredictionModel(self.activation, in_size, kernel, stride, self.output_size)
        self.dynamics = NomadDynamicsModel(self.activation, self.output_size)

        out_conv = SlimConv2d(
            out_channels,
            out_channels,
            kernel=1,
            stride=1,
            padding=None,
            activation_fn=None
        )

        self.representation = nn.Sequential(base_model._convs, out_conv)  # assumes you're using vision network not fc

        self.hidden = None
        self.last_action = None

    @override(MuZeroModel)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        pass

    @override(MuZeroModel)
    def reward_function(self) -> TensorType:
        pass

    @override(MuZeroModel)
    def representation_function(self, obs: TensorType) -> TensorType:
        pass
