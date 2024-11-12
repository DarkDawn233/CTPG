from typing import List, Tuple

import torch

from mtrl.agent.components.critic import QFunction, Critic
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import ConfigType, TensorType

class GuideCritic(Critic):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: int,
        hidden_dim: int,
        num_layers: int,
        guide_encoder_cfg: ConfigType,
        guide_multitask_cfg: ConfigType,
    ):
        """Critic component for the agent.

        Args:
            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            hidden_dim (int): hidden dimensionality of the actor.
            num_layers (int): number of layers in the actor.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        super(GuideCritic, self).__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_cfg=guide_encoder_cfg,
            multitask_cfg=guide_multitask_cfg,
        )

    def _make_qfunction(
        self,
        action_shape: List[int],
        hidden_dim: int,
        num_layers: int,
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ) -> QFunction:
        """Make the QFunction.

        Args:
            action_shape (List[int]):
            hidden_dim (int):
            num_layers (int):
            env_obs_shape (int):
            multitask_cfg (ConfigType):

        Returns:
            QFunction:
        """
        obs_dim = self._cal_obs_dim(encoder_cfg, multitask_cfg)
        
        return QFunction(
            input_dim=obs_dim,
            output_dim=action_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            multitask_cfg=multitask_cfg,
        )

    def forward(
        self,
        mtobs: MTObs,
        detach_encoder: bool = False,
    ) -> Tuple[TensorType, TensorType]:

        task_info = mtobs.task_info
        assert task_info is not None
        # detach_encoder allows to stop gradient propogation to encoder
        if self.should_condition_encoder_on_task_info:
            obs = self.encode(mtobs=mtobs, detach=detach_encoder)
        else:
            # making a new task_info since we do not want to condition on
            # # the task encoding.
            temp_task_info = TaskInfo(
                encoding=None,
                compute_grad=task_info.compute_grad,
                env_index=task_info.env_index,
            )
            temp_mtobs = MTObs(
                env_obs=mtobs.env_obs, task_obs=mtobs.task_obs, task_info=temp_task_info
            )
            obs = self.encode(mtobs=temp_mtobs, detach=detach_encoder)

        if self.use_task_onehot:
            onehot_encode = self.task_onehot_embedding.to(task_info.env_index.device)[task_info.env_index.squeeze(-1)]
            obs = torch.cat([obs, onehot_encode], dim=-1)

        mtobs_for_q = MTObs(
            env_obs=obs,
            task_obs=mtobs.task_obs,
            task_info=mtobs.task_info,
        )
        q1 = self.Q1(mtobs=mtobs_for_q)
        q2 = self.Q2(mtobs=mtobs_for_q)
        if self.should_use_multi_head_policy:
            q_mask = self.moe_masks.get_mask(task_info=task_info)
            sum_of_q_count = q_mask.sum(dim=0)
            sum_of_q1 = (q1 * q_mask).sum(dim=0)
            q1 = sum_of_q1 / sum_of_q_count
            sum_of_q2 = (q2 * q_mask).sum(dim=0)
            q2 = sum_of_q2 / sum_of_q_count

        return q1, q2