from typing import Dict, List

import gym
import torch
from ray.rllib import SampleBatch
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.torch.misc import same_padding, SlimConv2d, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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
            kernel,
            stride,
            None,
            activation_fn=self.activation)

        self.layer2 = SlimConv2d(
                    self.channels,
                    self.output_size,
                    [1, 1],
                    1,
                    padding,
                    activation_fn=None)

        self.policy = nn.Sequential(self.layer1, self.layer2, nn.Flatten())

        self.vlayer1 = SlimConv2d(
                self.channels,
                self.channels,
                kernel,
                stride,
                None,
                activation_fn=self.activation)

        self.vlayer2 = SlimConv2d(
                in_channels=self.channels,
                out_channels=1,
                kernel=1,
                stride=1,
                padding=None,
                activation_fn=None)

        self.value = nn.Sequential(self.vlayer1, self.vlayer2, nn.Flatten())

    def forward(self, hidden):
        """
        Hidden state should be Batch x 256 x ? x ? by default configs
        """
        policy_out = self.policy(hidden)

        value_out = self.value(hidden)

        return policy_out, value_out


class MuZeroDynamicsModel(nn.Module):
    def __init__(self, activation, channels=256):
        nn.Module.__init__(self)

        self.activation = activation
        self.channels = channels

        self.dynamic_layers = [
            SlimConv2d(
                self.channels + 1 if i == 0 else self.channels,  # encode actions for first layer needs extra channel
                self.channels,
                kernel=3,
                stride=1,
                padding=None,
                activation_fn=self.activation
            ) for i in range(10)
        ]

        self.dynamic_head = SlimConv2d(
            self.channels,
            self.channels,
            kernel=3,
            stride=1,
            padding=None,
            activation_fn=None
        )

        self.dynamic = nn.Sequential(*self.dynamic_layers)

        self.flatten = nn.Flatten()

        self.reward_layers = [
            SlimFC(
                256 * 6 * 6 if i == 0 else 256,
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
        if isinstance(action, int):
            action = torch.LongTensor(action).to(self.device)
        else:
            assert isinstance(action, torch.Tensor)
            action = action.long().to(self.device)

        new_tensor = torch.cat((hidden, action), dim=1)

        return new_tensor


class MuZeroModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, base_model: ModelV2):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        out_conv = SlimConv2d(
            256,
            256,
            kernel=3,
            stride=1,
            padding=None,
            activation_fn=None
        )

        self.representation = nn.Sequential(base_model._convs, out_conv)  # assumes you're using vision network not fully connected one
        self.activation = base_model.activation
        self.output_size = base_model.num_outputs
        self.prediction = MuZeroPredictionModel(self.activation, base_model.in_size, base_model.kernel,
                                                base_model.stride, self.output_size)
        self.dynamics = MuZeroDynamicsModel(self.activation)

        self.hidden = None
        self.last_action = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict[SampleBatch.OBS].float().permute(0, 3, 1, 2)
        representation = self.representation_function(obs)

        self.hidden = representation
        self.last_action = input_dict[SampleBatch.ACTIONS][-1]

    def value_function(self) -> TensorType:
        assert self.hidden is not None, "must call forward() first"

        _, value = self.prediction_function(self.hidden)

        return value

    def policy_function(self, current_obs: TensorType) -> (TensorType, TensorType):
        obs = current_obs.float().permute(0, 3, 1, 2)

        hidden = self.representation_function(obs)

        new_policy, _ = self.prediction_function(hidden)

        return new_policy

    def prediction_function(self, hidden: TensorType) -> (TensorType, TensorType):
        return self.prediction(hidden)

    def reward_function(self) -> TensorType:
        assert self.hidden is not None and self.last_action is not None, "must call forward() first"

        reward, _ = self.dynamics_function(self.hidden, self.last_action)

        return reward

    def dynamics_function(self, hidden: TensorType, action) -> (TensorType, TensorType):
        return self.dynamics(hidden, action)

    def representation_function(self, obs: TensorType) -> TensorType:
        return self.representation(obs)

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
