"""Implementation of the soft routing network and MLP described in
"Multi-Task Reinforcement Learning with Soft Modularization"
Reference: https://github.com/RchalYang/Soft-Module/blob/main/torchrl/networks/nets.py
"""


from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from mtrl.agent.components.base import Component as BaseComponent
from mtrl.agent.components.moe_layer import Linear
from mtrl.agent import utils as agent_utils
from mtrl.utils.types import TensorType

class SoftModularizedMLP(BaseComponent):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        task_num: int,
        hidden_features: int,
        hidden_layers: int,
        emb_hidden_layers: int,
        num_layers: int,
        num_modules: int,
        module_hidden_features: int,
        gating_hidden_features: int,
        cond_obs: bool = True,
        bias: bool = True,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        self.in_task_features = task_num
        self.in_obs_features = in_features - task_num
        self.cond_obs = cond_obs
        self.num_modules = num_modules

        self.obs_net = agent_utils.build_mlp(
            input_dim=self.in_obs_features,
            hidden_dim=hidden_features,
            num_layers=hidden_layers - 1,
            output_dim=hidden_features,
        )
        self.emb_net = agent_utils.build_mlp(
            input_dim=self.in_task_features,
            hidden_dim=hidden_features,
            num_layers=emb_hidden_layers - 1,
            output_dim=hidden_features,
        )
        self.emb_obs_activation = nn.ReLU()

        layers: List[nn.Module] = []
        current_in_features = hidden_features
        for _ in range(num_layers):
            linear = Linear(
                num_experts=num_modules,
                in_features=current_in_features,
                out_features=module_hidden_features,
                bias=bias,
            )
            layers.append(nn.Sequential(linear, nn.ReLU()))
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = module_hidden_features
        self.layers = nn.ModuleList(layers)
        self.last = nn.Linear(module_hidden_features, out_features, bias=bias)

        self.routing_network = RoutingNetwork(
            in_features=hidden_features,
            hidden_features=gating_hidden_features,
            num_modules_per_layer=num_modules,
            num_layers=num_layers
        )

    def forward(self, env_obs: TensorType) -> TensorType:
        env_obs, task_obs = torch.split(env_obs, [self.in_obs_features, self.in_task_features], dim=-1)
        obs_inp = self.obs_net(env_obs)
        routing_inp = self.emb_net(task_obs)
        if self.cond_obs:
            assert routing_inp.shape == obs_inp.shape
            routing_inp = routing_inp * obs_inp
        
        probs = self.routing_network(routing_inp)
        assert len(self.layers) == len(probs)

        obs_inp = obs_inp.unsqueeze(0).repeat(self.num_modules, 1, 1)
        for prob, layer in zip(probs, self.layers):
            obs_inp = layer(obs_inp).permute(1, 0, 2)
            obs_inp = (prob @ obs_inp).permute(1, 0, 2)
            
        obs_inp = obs_inp.squeeze(0)
        out = self.last(obs_inp)
        return out

class RoutingNetwork(BaseComponent):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_modules_per_layer: int,
        num_layers: int,
    ) -> None:
        """Class to implement the routing network in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        """
        super().__init__()

        self.num_modules_per_layer = num_modules_per_layer

        self.projection_before_routing = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features=in_features,
                out_features=hidden_features,
            )
        )
        
        self.W_d = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_features,
                    out_features=self.num_modules_per_layer ** 2,
                )
                for _ in range(num_layers - 1)
            ] + [
                nn.Linear(
                    in_features=hidden_features,
                    out_features=self.num_modules_per_layer,
                )
            ]
        )

        self.W_u = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.num_modules_per_layer ** 2,
                    out_features=hidden_features,
                )
                for _ in range(num_layers - 1)
            ]
        )  # the first layer does not need W_u

    def forward(self, emb_obs: TensorType) -> List[TensorType]:
        probs: List[TensorType] = []
        base_shape = emb_obs.shape[:-1]
        probs_shape = base_shape + torch.Size([self.num_modules_per_layer, self.num_modules_per_layer])
        last_prob_shape = base_shape + torch.Size([1, self.num_modules_per_layer])
        inp = self.projection_before_routing(emb_obs)

        p = self.W_d[0](F.relu(inp))
        probs.append(F.softmax(p.reshape(probs_shape), dim=-1))

        for W_u, W_d in zip(self.W_u[:-1], self.W_d[1:-1]):
            p = W_d(F.relu((W_u(p) * inp)))
            probs.append(F.softmax(p.reshape(probs_shape), dim=-1))
        
        p = self.W_d[-1](F.relu((self.W_u[-1](p) * inp)))
        probs.append(F.softmax(p.reshape(last_prob_shape), dim=-1))

        return probs

