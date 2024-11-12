# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
from copy import deepcopy
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from mtrl.agent import utils as agent_utils
from mtrl.agent.abstract import Agent as AbstractAgent
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.agent.optimizer import PCGrad
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.utils.types import ConfigType, ModelType, ParameterType, TensorType
from mtrl.utils.utils import mask_extreme_task


class Agent(AbstractAgent):
    """SAC algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
        logger: Logger = None,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )

        self.should_use_task_encoder = self.multitask_cfg.should_use_task_encoder

        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        self.actor = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape[0] * 2
        ).to(self.device)

        self.critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape[0]
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape[0]
        ).to(self.device)
        
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(init_temperature, dtype=np.float32)
                    for _ in range(self.num_envs)
                ]
            ).to(self.device)
        )
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self._components = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_alpha": self.log_alpha,  # type: ignore[dict-item]
        }
        # optimizers
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )
        self.should_use_pcgrad = self.multitask_cfg.should_use_pcgrad
        if self.should_use_pcgrad:
            self.actor_optimizer = PCGrad(self.actor_optimizer)
            self.critic_optimizer = PCGrad(self.critic_optimizer)
            self.log_alpha_optimizer = PCGrad(self.log_alpha_optimizer)

        if loss_reduction not in ["mean", "alpha_weight"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )

        self.loss_reduction = loss_reduction

        self._optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "log_alpha": self.log_alpha_optimizer,
        }

        if self.should_use_task_encoder:
            self.task_encoder = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.model_cfg,
            ).to(self.device)
            name = "task_encoder"
            self._components[name] = self.task_encoder
            self.task_encoder_optimizer = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.optimizer_cfg,
                params=self.get_parameters(name=name),
            )
            if self.should_use_pcgrad:
                self.task_encoder_optimizer = PCGrad(self.task_encoder_optimizer)
            self._optimizers[name] = self.task_encoder_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        if cfg_to_load_model:
            self.load(**cfg_to_load_model)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name != "log_alpha":
                component.train(training)

    def get_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.multitask_cfg.should_use_disentangled_alpha:
            return self.log_alpha[env_index].exp()
        else:
            return self.log_alpha[0].exp()

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(env_index.to(self.device))
        return self.task_encoder(env_index.to(self.device))

    def act(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        self.train(sample)
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, compute_grad=False, env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            return action.detach().cpu().numpy()

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=False)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=True)

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")

    def _compute_gradient(
        self,
        loss: TensorType,
        task_loss_list: List[TensorType],
        parameters: List[ParameterType],
        step: int,
        component_names: List[str],
        retain_graph: bool = False,
    ):
        """Method to override the gradient computation.

            Useful for algorithms like PCGrad and GradNorm.

        Args:
            loss (TensorType):
            parameters (List[ParameterType]):
            step (int): step for tracking the training of the agent.
            component_names (List[str]):
            retain_graph (bool, optional): if it should retain graph. Defaults to False.
        """
        if self.should_use_pcgrad:
            # Maybe error with task encoder
            for component_name in component_names:
                opt = self._optimizers[component_name]
                opt.pc_backward(task_loss_list)
        else:
            loss.backward(retain_graph=retain_graph)

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=batch.task_obs, task_info=task_info)
        _, policy_action, log_pi, _ = self.actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=policy_action)
        return (
            torch.min(target_Q1, target_Q2)
            - self.get_alpha(env_index=batch.task_obs).detach() * log_pi
        )

    def update_critic(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        with torch.no_grad():
            target_V = self._get_target_V(batch=batch, task_info=task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=batch.task_obs, task_info=task_info)
        current_Q1, current_Q2 = self.critic(
            mtobs=mtobs,
            action=batch.action,
            detach_encoder=False,
        )
        assert current_Q1.shape == target_Q.shape

        critic_loss_q1 = F.mse_loss(current_Q1, target_Q, reduce=False)
        critic_loss_q2 = F.mse_loss(current_Q2, target_Q, reduce=False)
        critic_loss = critic_loss_q1 + critic_loss_q2
        if step < self.multitask_cfg.mask_loss_step:
            use_loss_threshold = False
        else:
            use_loss_threshold = self.multitask_cfg.use_loss_threshold
        critic_loss, task_critic_loss, critic_mask = mask_extreme_task(critic_loss, batch.task_obs,
            self.multitask_cfg.num_envs, use_loss_threshold, self.multitask_cfg.mask_loss_threshold)

        if self.loss_reduction == "mean":
            critic_loss = critic_loss.mean()
        elif self.loss_reduction == "alpha_weight":
            critic_loss = critic_loss.reshape(self.multitask_cfg.num_envs, -1).mean(-1)
            assert self.loss_alpha_weight.shape == critic_loss.shape
            critic_loss = (self.loss_alpha_weight * critic_loss).mean()
        logger.log("train/critic_loss", critic_loss, step)
        if self.multitask_cfg.use_loss_threshold:
            mask_num = (1 - critic_mask.int()).sum()
            logger.log("train/critic_mask", mask_num, step)

        component_names = ["critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=critic_loss,
            task_loss_list=task_critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )

        # Optimize the critic
        torch.nn.utils.clip_grad_norm_(parameters, self.multitask_cfg.clip_grad_norm)
        self.critic_optimizer.step()

        return critic_mask

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
        critic_mask: TensorType = None,
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """

        # detach encoder, so we don't update it with the actor loss
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=batch.task_obs,
            task_info=task_info,
        )
        _, pi, log_pi, log_std = self.actor(mtobs=mtobs, detach_encoder=True)
        critic_task_info = self.get_task_info(task_info.encoding, compute_grad=False, env_index=task_info.env_index)
        critic_mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=batch.task_obs,
            task_info=critic_task_info,
        )
        actor_Q1, actor_Q2 = self.critic(mtobs=critic_mtobs, action=pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q)

        if step < self.multitask_cfg.mask_loss_step:
            use_loss_threshold = False
        else:
            use_loss_threshold = self.multitask_cfg.use_loss_threshold
        actor_loss, actor_task_loss, actor_mask = mask_extreme_task(actor_loss, batch.task_obs,
            self.multitask_cfg.num_envs, use_loss_threshold, self.multitask_cfg.mask_loss_threshold, mask = critic_mask)
        
        if self.loss_reduction == "mean":
            actor_loss = actor_loss.mean()
        elif self.loss_reduction == "alpha_weight":
            actor_loss = actor_loss.reshape(self.multitask_cfg.num_envs, -1).mean(-1)
            assert self.loss_alpha_weight.shape == actor_loss.shape
            actor_loss = (self.loss_alpha_weight * actor_loss).mean()
        
        logger.log("train/actor_loss", actor_loss, step)
        if self.multitask_cfg.use_loss_threshold:
            mask_num = (1 - actor_mask.int()).sum()
            logger.log("train/actor_mask", mask_num, step)

        logger.log("train/actor_target_entropy", self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        logger.log("train/actor_entropy", entropy.mean(), step)

        # optimize the actor
        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=actor_loss,
            task_loss_list=actor_task_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        torch.nn.utils.clip_grad_norm_(parameters, self.multitask_cfg.clip_grad_norm)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.get_alpha(batch.task_obs) * (-log_pi - self.target_entropy).detach())
        _, task_alpha_loss, _ = mask_extreme_task(alpha_loss, batch.task_obs, 
                self.multitask_cfg.num_envs, False)
        if self.loss_reduction == "mean" or self.loss_reduction == "alpha_weight":
            alpha_loss = alpha_loss.mean()
        logger.log("train/alpha_loss", alpha_loss, step)

        for index in range(self.multitask_cfg.num_envs):
            logger.log(f"train/alpha_{index}", self.get_alpha(torch.tensor(index)), step)
        
        self._compute_gradient(
            loss=alpha_loss,
            task_loss_list=task_alpha_loss,
            parameters=self.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        torch.nn.utils.clip_grad_norm_(self.get_parameters(name="log_alpha"), self.multitask_cfg.clip_grad_norm)
        self.log_alpha_optimizer.step()

        if "max_alpha" in self.multitask_cfg:
            max_log_alpha_data = torch.min(self.log_alpha, torch.ones_like(self.log_alpha) * np.log(self.multitask_cfg.max_alpha))
            self.log_alpha.data.copy_(max_log_alpha_data.data)

    def get_task_info(
        self, task_encoding: TensorType, compute_grad: bool, env_index: TensorType
    ) -> TaskInfo:
        """Encode task encoding into task info.

        Args:
            task_encoding (TensorType): encoding of the task.
            compute_grad (bool): keep gradient.
            env_index (TensorType): index of the environment.

        Returns:
            TaskInfo: TaskInfo object.
        """
        if self.should_use_task_encoder:
            task_encoding = task_encoding.detach() if not compute_grad else task_encoding
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=compute_grad, env_index=env_index
            )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info

    def update_transition_reward_model(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the transition model and reward decoder.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the task encoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        torch.nn.utils.clip_grad_norm_(self.get_parameters(name="task_encoder"), self.multitask_cfg.clip_grad_norm)
        self.task_encoder_optimizer.step()

    def update_decoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the decoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """

        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if buffer_index_to_sample is None:
            batch = replay_buffer.sample()
        else:
            batch = replay_buffer.sample(buffer_index_to_sample)

        if self.loss_reduction == "alpha_weight":
            softmax_temp = F.softmax(-self.log_alpha.detach())
            self.loss_alpha_weight = softmax_temp * self.num_envs

        logger.log("train/batch_reward", batch.reward.mean(), step)
        if self.should_use_task_encoder:
            self.task_encoder_optimizer.zero_grad()
            task_encoding = self.get_task_encoding(
                env_index=batch.task_obs.squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            task_encoding = None  # type: ignore[assignment]

        task_info = self.get_task_info(
            task_encoding=task_encoding,
            compute_grad=True,
            env_index=batch.task_obs,
        )

        critic_mask = self.update_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )
        if step % self.actor_update_freq == 0:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                compute_grad=True,
                env_index=batch.task_obs,
            )
            self.update_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
                critic_mask=critic_mask,
            )
        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if (
            "transition_model" in self._components
            and "reward_decoder" in self._components
        ):
            # some of the logic is a bit sketchy here. We will get to it soon.
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                compute_grad=True,
                env_index=batch.task_obs,
            )
            self.update_transition_reward_model(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if (
            "decoder" in self._components  # should_update_decoder
            and self.decoder is not None  # type: ignore[attr-defined]
            and step % self.decoder_update_freq == 0  # type: ignore[attr-defined]
        ):
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                compute_grad=True,
                env_index=batch.task_obs,
            )
            self.update_decoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        if self.should_use_task_encoder:
            self.update_task_encoder(
                batch=batch,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        
        # PaCo Trick
        if step > self.multitask_cfg.mask_loss_step:
            self._reset_task_while_extreme(critic_mask, logger, step)

        return batch.buffer_index

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]:
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            return list(self.actor.model.parameters())
        elif name in ["log_alpha", "alpha"]:
            return [self.log_alpha]
        elif name == "encoder":
            return list(self.critic.encoder.parameters())
        else:
            return list(self._components[name].parameters())
    
    def _reset_task_while_extreme(self, task_mask, logger, step):
        """
        PaCo Tricks
        """
        if self.should_use_task_encoder and self.multitask_cfg.actor_cfg.paco_should_use:
            extreme_task_mask = ~task_mask
            self.task_encoder.reset_weight_by_task_mask(extreme_task_mask)
