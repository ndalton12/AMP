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


class MuZeroPredictionModel(nn.Module):
    def __init__(self, activation, in_size, kernel, stride, output_size, channels=256):
        nn.Module.__init__(self)

        self.activation = activation
        self.channels = channels
        self.output_size = output_size

        in_size = [
            np.ceil((in_size[0] - kernel[0]) / stride),
            np.ceil((in_size[1] - kernel[1]) / stride)
        ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])

        self.layer1 = SlimConv2d(
            self.channels,
            self.channels,
            kernel=1,  # change to different value when representation function is changed?
            stride=1,
            padding=None,
            activation_fn=self.activation)

        self.layer2 = SlimConv2d(
                    self.channels,
                    self.output_size,
                    [1, 1],
                    1,
                    padding,
                    activation_fn=None)

        self.policy = nn.Sequential(self.layer1, self.layer2, nn.Flatten(), nn.Softmax())

        self.vlayer1 = SlimConv2d(
            self.channels,
            self.channels,
            kernel=1,  # change to different value when representation function is changed?
            stride=1,
            padding=None,
            activation_fn=self.activation)

        self.vlayer2 = SlimFC(
                self.channels,
                1,
                activation_fn=None)

        self.value = nn.Sequential(self.vlayer1, nn.Flatten(), self.vlayer2)

    def forward(self, hidden):
        """
        Hidden state should be Batch x 256 x 1 x 1 by default configs
        """
        policy_out = self.policy(hidden)

        value_out = self.value(hidden)

        return policy_out, value_out


class MuZeroDynamicsModel(nn.Module):
    def __init__(self, activation, action_size, channels=256):
        nn.Module.__init__(self)

        self.activation = activation
        self.channels = channels
        self.action_size = action_size

        self.dynamic_layers = [
            SlimConv2d(
                self.channels + self.action_size if i == 0 else self.channels,  # encode actions for first layer
                self.channels,
                kernel=1,
                stride=1,
                padding=None,
                activation_fn=self.activation
            ) for i in range(10)
        ]

        self.dynamic_head = SlimConv2d(
            self.channels,
            self.channels,
            kernel=1,
            stride=1,
            padding=None,
            activation_fn=None
        )

        self.dynamic = nn.Sequential(*self.dynamic_layers)

        self.flatten = nn.Flatten()

        self.reward_layers = [
            SlimFC(
                256 if i == 0 else 256,  # could make different later
                256 if i != 4 else 1,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation if i != 4 else None
            ) for i in range(5)
        ]

        self.reward_head = nn.Sequential(*self.reward_layers)

    def forward(self, hidden, action):
        input_tensor = self.encode(hidden, action)

        intermediate = self.dynamic(input_tensor)

        new_hidden = self.dynamic_head(intermediate)

        reward = self.reward_head(self.flatten(intermediate))

        return reward, new_hidden

    def encode(self, hidden, action):
        assert isinstance(action, torch.Tensor)

        action = torch.nn.functional.one_hot(action.long(), num_classes=self.action_size)
        action = action.unsqueeze(-1).unsqueeze(-1)  # action is now batch x space_size x 1 x 1

        new_tensor = torch.cat((hidden, action), dim=1)

        return new_tensor


class MuZeroModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, base_model: ModelV2):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.activation = base_model.model_config.get("conv_activation")
        self.output_size = base_model.num_outputs
        self.base_model = base_model

        filters = self.model_config["conv_filters"]
        out_channels, kernel, stride = filters[-1]
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]

        self.prediction = MuZeroPredictionModel(self.activation, in_size, kernel, stride, self.output_size)
        self.dynamics = MuZeroDynamicsModel(self.activation, self.output_size)

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

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        representation = self.representation_function(input_dict[SampleBatch.OBS])

        policy_logits, value = self.prediction_function(representation)

        return policy_logits, []  # state output must be a list here cuz modelv2 requires it rip

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.hidden is not None, "call forward() or representation() first"

        _, value = self.prediction_function(self.hidden)

        if len(value.shape) == 1:
            value = value.unsqueeze(0)  # add fake batch dim
        elif len(value.shape) == 0:
            value = value.unsqueeze(0).unsqueeze(0)  # add fake batch dim and real dim

        return value

    def policy_function(self, obs: TensorType) -> (TensorType, TensorType):
        hidden = self.representation_function(obs)

        new_policy, _ = self.prediction_function(hidden)

        return new_policy

    def prediction_function(self, hidden: TensorType) -> (TensorType, TensorType):
        return self.prediction(hidden)

    def reward_function(self, policy_logits) -> TensorType:
        assert self.hidden is not None, "must call forward() first"

        actions = torch.argmax(policy_logits, dim=1)
        reward, _ = self.dynamics_function(self.hidden, actions)

        return reward

    def dynamics_function(self, hidden: TensorType, action) -> (TensorType, TensorType):
        return self.dynamics(hidden, action)

    def representation_function(self, obs: TensorType) -> TensorType:
        obs = obs.float().permute(0, 3, 1, 2)
        output = self.representation(obs)
        self.hidden = output
        return output

    def metrics(self) -> Dict[str, TensorType]:
        return self.base_model.metrics()


def make_mu_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(
        action_space, config["model"], framework="torch")
    
    base_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework="torch")
    
    mu_model = MuZeroModel(obs_space, action_space, logit_dim, config["model"], name="MuZeroModel",
                           base_model=base_model)
    
    return mu_model
