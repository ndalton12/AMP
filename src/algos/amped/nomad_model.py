
import gym
import torch

from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from src.algos.mu_zero.mu_model import MuZeroModel, MuZeroPredictionModel


class NomadDynamicsModel(nn.Module):
    def __init__(self, activation, action_size, order, channels=256):
        nn.Module.__init__(self)

        self.activation = activation
        self.channels = channels
        self.action_size = action_size
        self.order = order

        self.dynamic_layers = [
            SlimConv2d(
                self.channels * self.order + self.action_size if i == 0 else self.channels,  # encode actions for first layer
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

    def forward(self, hiddens, action, evolving=False):
        if evolving:
            new_hiddens = [x.clone().detach() if i < len(hiddens) - 1 else x for i, x in enumerate(hiddens)]
        else:
            last = hiddens[-1]
            new_hiddens = [last.clone() for _ in range(self.order)]

        input_tensor = self.encode(new_hiddens, action)

        intermediate = self.dynamic(input_tensor)

        new_hidden = self.dynamic_head(intermediate)

        reward = self.reward_head(self.flatten(intermediate))

        return reward, new_hidden

    def encode(self, hiddens, action):
        assert isinstance(action, torch.Tensor)

        # hiddens is a list of order n of tensors batch x 256 x 1 x 1 (for now)

        # hack in case batch sizes are different for past hiddens
        # batch_sizes = [x.shape[0] for x in hiddens]
        # min_batch_size = min(batch_sizes)
        #
        # if any([x != min_batch_size for x in batch_sizes]):
        #     hiddens = [x[:min_batch_size] for x in hiddens]

        hidden = torch.cat(hiddens, dim=1)

        action = torch.nn.functional.one_hot(action.long(), num_classes=self.action_size)
        action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # action is now 1 x space_size x 1 x 1
        action = action.repeat(hidden.shape[0], 1, 1, 1).float()  # repeat along batch dim to match hidden

        new_tensor = torch.cat((hidden, action), dim=1)

        return new_tensor


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
        self.dynamics = NomadDynamicsModel(self.activation, self.output_size, self.order)

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
        self.cache = []

    @override(MuZeroModel)
    def dynamics_function(self, hidden: TensorType, action, evolving) -> (TensorType, TensorType):
        return self.dynamics(hidden, action, evolving)

    @override(MuZeroModel)
    def reward_function(self) -> TensorType:
        assert self.cache is not None and self.last_action is not None, "must call forward() first"

        reward, _ = self.dynamics_function(self.cache, self.last_action, evolving=False)

        return reward

    @override(MuZeroModel)
    def representation_function(self, obs: TensorType) -> TensorType:
        obs = obs.float().permute(0, 3, 1, 2)
        output = self.representation(obs)
        self.hidden = output

        if not self.cache:
            self.cache = [self.hidden] * self.order
        else:
            self.cache.append(self.hidden)
            self.cache.pop(0)

        return output


def make_nomad_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(
        action_space, config["model"], framework="torch")

    base_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework="torch")

    nomad_model = NomadModel(obs_space, action_space, logit_dim, config["model"], name="NomadModel",
                             base_model=base_model, order=config["mcts_param"]["order"])

    return nomad_model
