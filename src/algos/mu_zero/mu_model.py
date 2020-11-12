from typing import Dict, List

import gym
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


class MuZeroModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, base_model: ModelV2):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.base_model = base_model

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        pass

    def value_function(self) -> TensorType:
        pass

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
