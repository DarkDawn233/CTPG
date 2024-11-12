from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mtrl.agent.components import actor
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils.types import ConfigType, TensorType

class GuideActor(actor.Actor):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: int,
        hidden_dim: int,
        num_layers: int,
        log_std_bounds: Tuple[float, float],
        guide_encoder_cfg: ConfigType,
        guide_multitask_cfg: ConfigType,
    ):
        """Actor component for the agent.

        Args:
            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            hidden_dim (int): hidden dimensionality of the actor.
            num_layers (int): number of layers in the actor.
            log_std_bounds (Tuple[float, float]): bounds to clip log of standard deviation.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """

        super(GuideActor, self).__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            log_std_bounds=log_std_bounds,
            encoder_cfg=guide_encoder_cfg,
            multitask_cfg=guide_multitask_cfg,
        )

    def forward(
        self,
        mtobs: MTObs,
        detach_encoder: bool = False,
        other_info: dict = {},
        mask: TensorType = None,
    ) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        logits = self.model_forward(mtobs, detach_encoder, other_info)

        if mask is not None:
            _const_w = 1e20
            tmp = logits - _const_w * (1 - mask)
            tmp_max = tmp.max(dim=-1, keepdim=True)[0]
            tmp = torch.clamp(tmp - tmp_max, -_const_w, 1)
            probs = F.softmax(logits, dim=-1)
        else:
            logits_max = logits.max(dim=-1, keepdim=True)[0]
            logits = logits - logits_max
            probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        argmax_a = torch.argmax(probs, dim=-1)
        sample_a = dist.sample()

        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)

        return argmax_a, sample_a, probs, log_probs