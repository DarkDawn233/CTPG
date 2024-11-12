# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Component to encode the task."""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.utils.types import ConfigType, TensorType


class TaskEncoder(base_component.Component):
    def __init__(
        self,
        pretrained_embedding_cfg: ConfigType,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        """Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
        """
        self.num_embeddings = num_embeddings
        super().__init__()
        if pretrained_embedding_cfg.should_use:
            with open(pretrained_embedding_cfg.path_to_load_from) as f:
                metadata = json.load(f)
            ordered_task_list = pretrained_embedding_cfg.ordered_task_list
            pretrained_embedding = torch.Tensor(
                [metadata[task] for task in ordered_task_list]
            )
            assert num_embeddings == pretrained_embedding.shape[0]
            pretrained_embedding_dim = pretrained_embedding.shape[1]
            pretrained_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding,
                freeze=True,
            )
            projection_layer = nn.Sequential(
                nn.Linear(
                    in_features=pretrained_embedding_dim, out_features=2 * embedding_dim
                ),
                nn.ReLU(),
                nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
                nn.ReLU(),
            )
            projection_layer.apply(agent_utils.weight_init)
            self.embedding = nn.Sequential(  # type: ignore [call-overload]
                pretrained_embedding,
                nn.ReLU(),
                projection_layer,
            )

        else:
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
            self.embedding.apply(agent_utils.weight_init)
        if num_layers >= 0:
            self.use_trunk = True
            self.trunk = agent_utils.build_mlp(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )
            self.trunk.apply(agent_utils.weight_init)
        else:
            self.use_trunk = False

    def forward(self, env_index: TensorType) -> TensorType:
        if self.use_trunk:
            return self.trunk(self.embedding(env_index))
        else:
            return self.embedding(env_index)
    
    def reset_weight_by_task_mask(self, task_mask, valid_mask=None):
        """
        task_mask:
            True: Reset weight
            False: Keep weight
        """
        if not task_mask.any():
            return
        device = self.embedding.weight.device
        reset_task_set = torch.arange(self.num_embeddings, device=device)[task_mask]
        normal_task_mask = ~task_mask
        if valid_mask is not None:
            normal_task_mask = normal_task_mask & valid_mask.to(torch.bool)
        normal_task_set = torch.arange(self.num_embeddings, device=device)[normal_task_mask]
        new_weight = self.embedding.weight.data.clone()
        if len(normal_task_set) != 0:
            normal_task_weight = new_weight[normal_task_set, :]                                     # [n-x, p]
            interpolate_weight = torch.rand((reset_task_set.shape[0], normal_task_set.shape[0]), device=device)    # [x, n-x]
            interpolate_weight = F.normalize(interpolate_weight, p=1, dim=-1)
            new_weight_reset = interpolate_weight @ normal_task_weight  # [x, n-x] @ [n-x, p] -> [x, p]
        else:
            new_weight_reset = torch.rand_like(new_weight[reset_task_set, :])
            nn.init.xavier_uniform_(new_weight_reset)
        new_weight[reset_task_set, :] = new_weight_reset
        with torch.no_grad():
            self.embedding.weight.copy_(new_weight)
