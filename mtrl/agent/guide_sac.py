# Reference: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
from typing import Any, Dict, List, Optional, Tuple

from copy import deepcopy
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from mtrl.agent import sac
from mtrl.agent import utils as agent_utils
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.agent.optimizer import PCGrad
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.env.types import ObsType
from mtrl.utils.types import ConfigType, ModelType, ParameterType, TensorType
from mtrl.utils.utils import mask_extreme_task


class Agent(sac.Agent):
    """SAC algorithm + Guide policy"""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        guide_actor_cfg: ConfigType,
        guide_critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        guide_alpha_optimizer_cfg: ConfigType,
        guide_actor_optimizer_cfg: ConfigType,
        guide_critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        guide_multitask_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        guide_init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        guide_actor_update_freq: int,
        guide_critic_tau: float,
        guide_critic_target_update_freq: int,
        guide_encoder_tau: float,
        guide_hindsight: bool,
        loss_reduction: str = "mean",
        guide_loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
        logger: Logger = None,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            alpha_optimizer_cfg=alpha_optimizer_cfg,
            actor_optimizer_cfg=actor_optimizer_cfg,
            critic_optimizer_cfg=critic_optimizer_cfg,
            multitask_cfg=multitask_cfg,
            discount=discount,
            init_temperature=init_temperature,
            actor_update_freq=actor_update_freq,
            critic_tau=critic_tau,
            critic_target_update_freq=critic_target_update_freq,
            encoder_tau=encoder_tau,
            loss_reduction=loss_reduction,
            cfg_to_load_model=None,
            should_complete_init=False,
            logger=logger,
        )
        self.guide_multitask_cfg = guide_multitask_cfg

        self.guide_should_use_task_encoder = self.guide_multitask_cfg.should_use_task_encoder

        self.discount = discount
        self.guide_critic_tau = guide_critic_tau
        self.guide_encoder_tau = guide_encoder_tau
        self.guide_actor_update_freq = guide_actor_update_freq
        self.guide_critic_target_update_freq = guide_critic_target_update_freq

        self.guide_hindsight = guide_hindsight

        self.guide_actor = hydra.utils.instantiate(
            guide_actor_cfg, env_obs_shape=env_obs_shape, action_shape=multitask_cfg.num_envs
        ).to(self.device)

        self.guide_critic = hydra.utils.instantiate(
            guide_critic_cfg, env_obs_shape=env_obs_shape, action_shape=multitask_cfg.num_envs
        ).to(self.device)

        self.guide_critic_target = hydra.utils.instantiate(
            guide_critic_cfg, env_obs_shape=env_obs_shape, action_shape=multitask_cfg.num_envs
        ).to(self.device)

        if self.guide_multitask_cfg.use_comp:
            self.guide_critic_comp = hydra.utils.instantiate(
                guide_critic_cfg, env_obs_shape=env_obs_shape, action_shape=multitask_cfg.num_envs
            ).to(self.device)

            self.guide_critic_comp_target = hydra.utils.instantiate(
                guide_critic_cfg, env_obs_shape=env_obs_shape, action_shape=multitask_cfg.num_envs
            ).to(self.device)

        self.guide_log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(guide_init_temperature, dtype=np.float32)
                    for _ in range(self.num_envs)
                ]
            ).to(self.device)
        )
        self.guide_target_entropy = -np.log((1.0 / multitask_cfg.num_envs)) * 0.98

        self._components.update({
            "guide_actor": self.guide_actor,
            "guide_critic": self.guide_critic,
            "guide_critic_target": self.guide_critic_target,
            "guide_log_alpha": self.guide_log_alpha,  
        })
        if self.guide_multitask_cfg.use_comp:
            self._components.update({
                "guide_critic_comp": self.guide_critic_comp,
                "guide_critic_comp_target": self.guide_critic_comp_target,
            })

        # optimizers
        self.guide_actor_optimizer = hydra.utils.instantiate(
            guide_actor_optimizer_cfg, params=self.get_parameters(name="guide_actor")
        )
        self.guide_critic_optimizer = hydra.utils.instantiate(
            guide_critic_optimizer_cfg, params=self.get_parameters(name="guide_critic")
        )
        if self.guide_multitask_cfg.use_comp:
            self.guide_critic_comp_optimizer = hydra.utils.instantiate(
                guide_critic_optimizer_cfg, params=self.get_parameters(name="guide_critic_comp")
            )
        self.guide_log_alpha_optimizer = hydra.utils.instantiate(
            guide_alpha_optimizer_cfg, params=self.get_parameters(name="guide_log_alpha")
        )
        if guide_loss_reduction not in ["mean", "alpha_weight"]:
            raise ValueError(
                f"{guide_loss_reduction} is not a supported value for `guide_loss_reduction`."
            )
        self.guide_should_use_pcgrad = self.guide_multitask_cfg.should_use_pcgrad
        if self.guide_should_use_pcgrad:
            self.guide_actor_optimizer = PCGrad(self.guide_actor_optimizer)
            self.guide_critic_optimizer = PCGrad(self.guide_critic_optimizer)
            if self.guide_multitask_cfg.use_comp:
                self.guide_critic_comp_optimizer = PCGrad(self.guide_critic_comp_optimizer)
            self.guide_log_alpha_optimizer = PCGrad(self.guide_log_alpha_optimizer)

        self.guide_loss_reduction = guide_loss_reduction

        self._optimizers.update({
            "guide_actor": self.guide_actor_optimizer,
            "guide_critic": self.guide_critic_optimizer,
            "guide_log_alpha": self.guide_log_alpha_optimizer,
        })
        if self.guide_multitask_cfg.use_comp:
            self._optimizers.update({
                "guide_critic_comp": self.guide_critic_comp_optimizer, 
            })

        if self.guide_should_use_task_encoder:
            self.guide_task_encoder = hydra.utils.instantiate(
                self.guide_multitask_cfg.task_encoder_cfg.model_cfg,
            ).to(self.device)
            name = "guide_task_encoder"
            self._components[name] = self.guide_task_encoder
            self.guide_task_encoder_optimizer = hydra.utils.instantiate(
                self.guide_multitask_cfg.task_encoder_cfg.optimizer_cfg,
                params=self.get_parameters(name=name),
            )
            if self.guide_should_use_pcgrad:
                self.guide_task_encoder_optimizer = PCGrad(self.guide_task_encoder_optimizer)
            self._optimizers[name] = self.guide_task_encoder_optimizer

            if self.guide_multitask_cfg.use_comp:
                self.guide_task_encoder_comp = hydra.utils.instantiate(
                    self.guide_multitask_cfg.task_encoder_cfg.model_cfg,
                ).to(self.device)
                name = "guide_task_encoder_comp"
                self._components[name] = self.guide_task_encoder_comp
                self.guide_task_encoder_comp_optimizer = hydra.utils.instantiate(
                    self.guide_multitask_cfg.task_encoder_cfg.optimizer_cfg,
                    params=self.get_parameters(name=name),
                )
                if self.guide_should_use_pcgrad:
                    self.guide_task_encoder_comp_optimizer = PCGrad(self.guide_task_encoder_comp_optimizer)
                self._optimizers[name] = self.guide_task_encoder_comp_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)
        
        self.no_guide_prob = torch.zeros((self.guide_multitask_cfg.num_envs), device=device)
        self.no_guide = torch.zeros((self.guide_multitask_cfg.num_envs), dtype=torch.int64, device=device)
        self.guide_sample_scale = np.ones(self.guide_multitask_cfg.num_envs) / self.guide_multitask_cfg.num_envs

        self.guide_compare = self.guide_multitask_cfg.guide_compare

        self.use_no_guide = self.guide_multitask_cfg.use_no_guide

        self.guide_step = self.guide_multitask_cfg.guide_step
        self.guide_policy = -torch.ones((self.guide_multitask_cfg.num_envs), dtype=torch.int64, device=device)
        self.guide_step_already = torch.zeros((self.guide_multitask_cfg.num_envs), dtype=torch.int64, device=device)

        self.train_step = 0
    
    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        super().complete_init(cfg_to_load_model)

        self.guide_critic_target.load_state_dict(self.guide_critic.state_dict())
        self.guide_actor.encoder.copy_conv_weights_from(self.guide_critic.encoder)

        if self.guide_multitask_cfg.use_comp:
            self.guide_critic_comp_target.load_state_dict(self.guide_critic_comp.state_dict())


    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name not in ["log_alpha", "guide_log_alpha"]:
                component.train(training)
    
    def get_guide_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.guide_multitask_cfg.should_use_disentangled_alpha:
            return self.guide_log_alpha[env_index].exp()
        else:
            return self.guide_log_alpha[0].exp()
        
    def get_guide_task_encoding(
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
                return self.guide_task_encoder(env_index.to(self.device))
        return self.guide_task_encoder(env_index.to(self.device))

    def get_guide_task_encoding_comp(
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
                return self.guide_task_encoder_comp(env_index.to(self.device))
        return self.guide_task_encoder_comp(env_index.to(self.device))
    
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
        if not sample:
            return super().act(multitask_obs, modes, sample)
        else:
            return self.guide_sample(multitask_obs, modes, sample)
    
    def guide_sample(
        self,
        multitask_obs: ObsType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        self.train(sample)
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            change_guide = (self.guide_step_already >= self.guide_step).long()
            self.guide_step_already = (1 - change_guide) * self.guide_step_already
            self.guide_step_already += 1
            self.guide_policy = self.guide_policy * (1 - change_guide) + (-1) * change_guide
            new_guide_policy = (self.guide_policy == -1).long()
            other_info = {}
            if new_guide_policy.any():
                if self.guide_should_use_task_encoder:
                    guide_task_encoding = self.get_guide_task_encoding(
                        env_index=env_index, modexs=modes, disable_grad=True
                    )
                else:
                    guide_task_encoding = None  # type: ignore[assignment]
                guide_task_info = self.get_guide_task_info(
                    task_encoding=guide_task_encoding, compute_grad=False, env_index=env_index
                )
                guide_obs = env_obs.float().to(self.device)
                if len(guide_obs.shape) == 1 or len(guide_obs.shape) == 3:
                    guide_obs = guide_obs.unsqueeze(0)
                guide_mtobs = MTObs(env_obs=guide_obs, task_obs=env_index, task_info=guide_task_info)

                # Compare with V-value
                if self.guide_compare and self.train_step > self.guide_multitask_cfg.no_compare_step:
                    # Src act
                    if self.should_use_task_encoder:
                        src_task_encoding = self.get_task_encoding(
                            env_index=env_index, modes=modes, disable_grad=True
                        )
                    else:
                        src_task_encoding = None
                    src_task_info = self.get_task_info(
                        task_encoding=src_task_encoding, compute_grad=False, env_index=env_index
                    )
                    src_obs = env_obs.float().to(self.device)
                    if len(src_obs.shape) == 1 or len(src_obs.shape) == 3:
                        src_obs = src_obs.unsqueeze(0)  # Make a batch
                    src_mtobs = MTObs(env_obs=src_obs, task_obs=env_index, task_info=src_task_info)
                    _, src_pi, src_log_pi, _, src_dis = self.actor(mtobs=src_mtobs, dis_f = True)

                    src_pi_list = [src_pi]
                    src_log_pi_list = [src_log_pi]
                    for _ in range(1, self.guide_multitask_cfg.v_sample_time):
                        new_src_pi, new_z = src_dis.rsample(return_pretanh_value=True)
                        new_src_log_pi = src_dis.log_prob(new_src_pi, pre_tanh_value=new_z)
                        src_pi_list.append(new_src_pi)
                        src_log_pi_list.append(new_src_log_pi)
                    src_pi = torch.stack(src_pi_list, dim=0)
                    src_log_pi = torch.stack(src_log_pi_list, dim=0)
                    src_pi = src_pi.clamp(*self.action_range)

                    # Src V
                    src_pi = src_pi.reshape(-1, *src_pi.shape[2:])
                    c_env_index = env_index.unsqueeze(0).repeat(self.guide_multitask_cfg.v_sample_time, 1).reshape(-1)
                    if self.should_use_task_encoder:
                        src_c_task_encoding = self.get_task_encoding(
                            env_index=c_env_index, modes=modes, disable_grad=True
                        )
                    else:
                        src_c_task_encoding = None
                    src_c_task_info = self.get_task_info(
                        task_encoding=src_c_task_encoding, compute_grad=False, env_index=c_env_index
                    )
                    src_c_obs = src_obs.unsqueeze(0).repeat(self.guide_multitask_cfg.v_sample_time, 1, 1).reshape(-1, *src_obs.shape[1:])
                    src_c_mtobs = MTObs(env_obs=src_c_obs, task_obs=c_env_index, task_info=src_c_task_info)
                    src_q1, src_q2 = self.critic(mtobs=src_c_mtobs, action=src_pi)
                    src_q1 = src_q1.reshape(self.guide_multitask_cfg.v_sample_time, -1, *src_q1.shape[1:]).mean(0)
                    src_q2 = src_q2.reshape(self.guide_multitask_cfg.v_sample_time, -1, *src_q2.shape[1:]).mean(0)
                    src_q = torch.mean(torch.stack([src_q1, src_q2], -1), -1) - self.get_alpha(env_index=env_index).unsqueeze(-1) * src_log_pi.mean(0)

                    # Guide Q
                    if self.guide_multitask_cfg.use_comp:
                        guide_q1, guide_q2 = self.guide_critic_comp(mtobs=guide_mtobs)
                    else:
                        guide_q1, guide_q2 = self.guide_critic(mtobs=guide_mtobs)
                    guide_q = torch.mean(torch.stack([guide_q1, guide_q2], -1), -1)
                    
                    guide_mask = (guide_q >= src_q).float()
                    use_guide = 1 - (guide_mask.sum(-1) == 0).long()
                else:
                    guide_mask = None
                    use_guide = torch.ones_like(env_index)

                # Guide act 
                argmax_guide_a, sample_guide_a, _, _ = self.guide_actor(mtobs=guide_mtobs, mask=guide_mask)
                sample_guide_a = sample_guide_a if sample else argmax_guide_a
                assert sample_guide_a.shape == env_index.shape
                sample_guide_a = env_index * (1 - use_guide) + sample_guide_a * use_guide

                self.guide_policy = self.guide_policy * (1 - new_guide_policy) + sample_guide_a * new_guide_policy

            sample_guide_a = self.guide_policy
            guide_now = 1 - self.no_guide
            sample_guide_a = env_index * (1 - guide_now) + sample_guide_a * guide_now
            other_info["guide_begin"] = new_guide_policy.detach().cpu().numpy().astype(np.int32)
            other_info["guide_action"] = sample_guide_a.detach().cpu().numpy()
            
            # Agent act
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=sample_guide_a, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, compute_grad=False, env_index=sample_guide_a
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=sample_guide_a, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            action = pi if sample else mu
            
            action = action.clamp(*self.action_range)
            return action.detach().cpu().numpy(), other_info


    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",

            "guide_actor",
            "guide_critic",
            "guide_critic_comp",
            "guide_transition_model",
            "guide_reward_decoder",
            "guide_decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder",
                              "guide_log_alpha", "guide_encoder", "guide_task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")
    
    def _compute_guide_gradient(
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
        if self.guide_should_use_pcgrad:
            # Maybe error with task encoder
            for component_name in component_names:
                opt = self._optimizers[component_name]
                opt.pc_backward(task_loss_list)
        else:
            loss.backward(retain_graph=retain_graph)

    def _get_guide_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs[:,-1], task_obs=batch.task_obs[:,0], task_info=task_info)
        _, _, probs, log_probs = self.guide_actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.guide_critic_target(mtobs=mtobs)
        assert target_Q1.shape == probs.shape
        target_V = probs * (torch.min(target_Q1, target_Q2) - 
                self.get_guide_alpha(env_index=batch.task_obs[:,0]).detach() * log_probs)
        target_V = torch.sum(target_V, dim=-1, keepdim=True)
        return target_V

    def _get_guide_comp_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo, task_info_comp: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs[:,-1], task_obs=batch.task_obs[:,0], task_info=task_info)
        _, _, probs, log_probs = self.guide_actor(mtobs=mtobs)
        mtobs_comp = MTObs(env_obs=batch.next_env_obs[:,-1], task_obs=batch.task_obs[:,0], task_info=task_info_comp)
        target_Q1, target_Q2 = self.guide_critic_comp_target(mtobs=mtobs_comp)
        assert target_Q1.shape == probs.shape

        if self.should_use_task_encoder:
            pi_task_encoding = self.get_task_encoding(
                env_index=batch.task_obs[:,0].squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            pi_task_encoding = None  # type: ignore[assignment]

        pi_task_info = self.get_task_info(
            task_encoding=pi_task_encoding,
            compute_grad=False,
            env_index=batch.task_obs[:,0],
        )
        pi_mtobs = MTObs(env_obs=batch.next_env_obs[:,-1], task_obs=batch.task_obs[:,0], task_info=pi_task_info)
        _, pi_action, pi_log_pi, _ = self.actor(mtobs=pi_mtobs)
        target_V = probs * (torch.min(target_Q1, target_Q2))
        target_V = torch.sum(target_V, dim=-1, keepdim=True)
        return target_V
    
    def update_guide_critic(
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
            guide_target_V = self._get_guide_target_V(batch=batch, task_info=task_info)
            guide_target_Q = batch.reward[:,-1] + (batch.not_done[:,-1] * self.discount * guide_target_V)
            for i in range(self.guide_step-2, -1, -1):
                guide_target_Q = batch.reward[:,i] + batch.not_done[:,i] * self.discount * guide_target_Q


        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs[:,0], task_obs=None, task_info=task_info)
        guide_current_Q1, guide_current_Q2 = self.guide_critic(
            mtobs=mtobs,
            detach_encoder=False,
        )
        guide_current_Q1 = guide_current_Q1.gather(-1, batch.guide_action.long())
        guide_current_Q2 = guide_current_Q2.gather(-1, batch.guide_action.long())
        assert guide_current_Q1.shape == guide_target_Q.shape
        
        guide_critic_loss_q1 = F.mse_loss(guide_current_Q1, guide_target_Q, reduce=False)
        guide_critic_loss_q2 = F.mse_loss(guide_current_Q2, guide_target_Q, reduce=False)
        guide_critic_loss = guide_critic_loss_q1 + guide_critic_loss_q2

        if step < self.guide_multitask_cfg.mask_loss_step:
            use_loss_threshold = False
        else:
            use_loss_threshold = self.guide_multitask_cfg.use_loss_threshold
        guide_critic_loss, guide_task_critic_loss, guide_critic_mask = mask_extreme_task(guide_critic_loss, batch.task_obs[:,0], 
            self.guide_multitask_cfg.num_envs, use_loss_threshold, self.guide_multitask_cfg.mask_loss_threshold)

        if self.guide_loss_reduction == "mean":
            guide_critic_loss = guide_critic_loss.mean()
        elif self.guide_loss_reduction == "alpha_weight":
            guide_critic_loss = guide_critic_loss.reshape(self.guide_multitask_cfg.num_envs, -1).mean(-1)
            assert self.guide_loss_alpha_weight.shape == guide_critic_loss.shape
            guide_critic_loss = (self.guide_loss_alpha_weight * guide_critic_loss).mean()
        logger.log("train/guide_critic_loss", guide_critic_loss, step)
        if self.guide_multitask_cfg.use_loss_threshold:
            mask_num = (1 - guide_critic_mask.int()).sum()
            logger.log("train/guide_critic_mask", mask_num, step)

        component_names = ["guide_critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("guide_task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("guide_task_encoder")

        self._compute_guide_gradient(
            loss=guide_critic_loss,
            task_loss_list=guide_task_critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )

        # Optimize the critic
        torch.nn.utils.clip_grad_norm_(parameters, self.guide_multitask_cfg.clip_grad_norm)
        self.guide_critic_optimizer.step()

        return guide_critic_mask

    def update_guide_critic_comp(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        task_info_comp: TaskInfo,
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
            guide_comp_target_V = self._get_guide_comp_target_V(batch=batch, task_info=task_info, task_info_comp=task_info_comp)
            guide_comp_target_Q = guide_comp_target_V
            if self.should_use_task_encoder:
                pi_task_encoding = self.get_task_encoding(
                    env_index=batch.task_obs[:,0].squeeze(1),
                    disable_grad=False,
                    modes=["train"],
                )
            else:
                pi_task_encoding = None  # type: ignore[assignment]
            pi_task_info = self.get_task_info(
                task_encoding=pi_task_encoding,
                compute_grad=False,
                env_index=batch.task_obs[:,0],
            )

            for i in range(self.guide_step-1, -1, -1):
                pi_mtobs = MTObs(env_obs=batch.env_obs[:,i], task_obs=batch.task_obs[:,0], task_info=pi_task_info)
                _, pi_action, pi_log_pi, _ = self.actor(mtobs=pi_mtobs)
                guide_comp_target_Q = batch.reward[:,i] - self.get_alpha(env_index=batch.task_obs[:,0]).detach() * pi_log_pi + \
                                        batch.not_done[:,i-1] * self.discount * guide_comp_target_Q

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs[:,0], task_obs=None, task_info=task_info_comp)
        guide_comp_current_Q1, guide_comp_current_Q2 = self.guide_critic_comp(
            mtobs=mtobs,
            detach_encoder=False,
        )
        guide_comp_current_Q1 = guide_comp_current_Q1.gather(-1, batch.guide_action.long())
        guide_comp_current_Q2 = guide_comp_current_Q2.gather(-1, batch.guide_action.long())
        assert guide_comp_current_Q1.shape == guide_comp_target_Q.shape
        
        guide_critic_comp_loss_q1 = F.mse_loss(guide_comp_current_Q1, guide_comp_target_Q, reduce=False)
        guide_critic_comp_loss_q2 = F.mse_loss(guide_comp_current_Q2, guide_comp_target_Q, reduce=False)
        guide_critic_comp_loss = guide_critic_comp_loss_q1 + guide_critic_comp_loss_q2

        if step < self.guide_multitask_cfg.mask_loss_step:
            use_loss_threshold = False
        else:
            use_loss_threshold = self.guide_multitask_cfg.use_loss_threshold
        guide_critic_comp_loss, guide_task_critic_comp_loss, guide_critic_comp_mask = mask_extreme_task(guide_critic_comp_loss, batch.task_obs[:,0], 
            self.guide_multitask_cfg.num_envs, use_loss_threshold, self.guide_multitask_cfg.mask_loss_threshold)

        if self.guide_loss_reduction == "mean":
            guide_critic_comp_loss = guide_critic_comp_loss.mean()
        elif self.guide_loss_reduction == "alpha_weight":
            guide_critic_comp_loss = guide_critic_comp_loss.reshape(self.guide_multitask_cfg.num_envs, -1).mean(-1)
            assert self.guide_loss_alpha_weight.shape == guide_critic_comp_loss.shape
            guide_critic_comp_loss = (self.guide_loss_alpha_weight * guide_critic_comp_loss).mean()
        logger.log("train/guide_critic_comp_loss", guide_critic_comp_loss, step)
        if self.guide_multitask_cfg.use_loss_threshold:
            mask_num = (1 - guide_critic_comp_mask.int()).sum()
            logger.log("train/guide_critic_comp_mask", mask_num, step)

        component_names = ["guide_critic_comp"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info_comp.compute_grad:
            component_names.append("guide_task_encoder_comp")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("guide_task_encoder_comp")

        self._compute_guide_gradient(
            loss=guide_critic_comp_loss,
            task_loss_list=guide_task_critic_comp_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )

        # Optimize the critic
        torch.nn.utils.clip_grad_norm_(parameters, self.guide_multitask_cfg.clip_grad_norm)
        self.guide_critic_comp_optimizer.step()

        return guide_critic_comp_mask

    def update_guide_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
        guide_critic_mask: TensorType = None,
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
            env_obs=batch.env_obs[:,0],
            task_obs=batch.task_obs[:,0],
            task_info=task_info,
        )
        _, _, probs, log_probs = self.guide_actor(mtobs=mtobs, detach_encoder=True)

        guide_actor_Q1, guide_actor_Q2 = self.guide_critic(
            mtobs=mtobs, detach_encoder=True
        )
        assert guide_actor_Q1.shape == probs.shape
        min_actor_Q = torch.min(guide_actor_Q1, guide_actor_Q2)
        inside_term = self.get_guide_alpha(batch.task_obs[:,0]).detach() * log_probs - min_actor_Q
        guide_actor_loss = torch.sum(probs * inside_term, dim=-1, keepdim=True)

        if step < self.guide_multitask_cfg.mask_loss_step:
            use_loss_threshold = False
        else:
            use_loss_threshold = self.guide_multitask_cfg.use_loss_threshold
        guide_actor_loss, guide_task_actor_loss, guide_actor_mask = mask_extreme_task(guide_actor_loss, batch.task_obs[:,0],
            self.guide_multitask_cfg.num_envs, use_loss_threshold, self.guide_multitask_cfg.mask_loss_threshold, mask=guide_critic_mask)
        
        if self.guide_loss_reduction == "mean":
            guide_actor_loss = guide_actor_loss.mean()
        elif self.guide_loss_reduction == "alpha_weight":
            guide_actor_loss = guide_actor_loss.reshape(self.guide_multitask_cfg.num_envs, -1).mean(-1)
            assert self.guide_loss_alpha_weight.shape == guide_actor_loss.shape
            guide_actor_loss = (self.guide_loss_alpha_weight * guide_actor_loss).mean()

        logger.log("train/guide_actor_loss", guide_actor_loss, step)
        if self.guide_multitask_cfg.use_loss_threshold:
            mask_num = (1 - guide_actor_mask.int()).sum()
            logger.log("train/guide_actor_mask", mask_num, step)

        logger.log("train/guide_actor_target_entropy", self.guide_target_entropy, step)
        entropy = - torch.sum(probs * log_probs, dim=-1, keepdim=True)
        logger.log("train/guide_actor_entropy", entropy.mean(), step)

        # optimize the actor
        component_names = ["guide_actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("guide_task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("guide_task_encoder")

        self._compute_guide_gradient(
            loss=guide_actor_loss,
            task_loss_list=guide_task_actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        torch.nn.utils.clip_grad_norm_(parameters, self.guide_multitask_cfg.clip_grad_norm)
        self.guide_actor_optimizer.step()

        self.guide_log_alpha_optimizer.zero_grad()
        guide_alpha_loss = (self.get_guide_alpha(batch.task_obs[:, 0])
                            * (entropy - self.guide_target_entropy).detach())
        _, guide_task_alpha_loss, _ = mask_extreme_task(guide_alpha_loss, batch.task_obs[:, 0], 
                self.multitask_cfg.num_envs, False)
            
        if self.guide_loss_reduction == "mean" or self.guide_loss_reduction == "alpha_weight":
            guide_alpha_loss = guide_alpha_loss.mean()
        logger.log("train/guide_alpha_loss", guide_alpha_loss, step)

        for index in range(self.guide_multitask_cfg.num_envs):
            logger.log(f"train/guide_alpha_{index}", self.get_guide_alpha(torch.tensor(index)), step)

        self._compute_guide_gradient(
            loss=guide_alpha_loss,
            task_loss_list=guide_task_alpha_loss,
            parameters=self.get_parameters(name="guide_log_alpha"),
            step=step,
            component_names=["guide_log_alpha"],
            **kwargs_to_compute_gradient,
        )
        torch.nn.utils.clip_grad_norm_(self.get_parameters(name="guide_log_alpha"), self.guide_multitask_cfg.clip_grad_norm)
        self.guide_log_alpha_optimizer.step()

        if "max_alpha" in self.guide_multitask_cfg:
            max_guide_log_alpha_data = torch.min(self.guide_log_alpha, torch.ones_like(self.guide_log_alpha) * np.log(self.guide_multitask_cfg.max_alpha))
            self.guide_log_alpha.data.copy_(max_guide_log_alpha_data.data)
    
    def get_guide_task_info(
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

        if self.guide_should_use_task_encoder:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=compute_grad, env_index=env_index
            )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info
    
    def update_guide_task_encoder(
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
        torch.nn.utils.clip_grad_norm_(self.get_parameters(name="guide_task_encoder"), self.guide_multitask_cfg.clip_grad_norm)
        self.guide_task_encoder_optimizer.step()
        if self.guide_multitask_cfg.use_comp:
            torch.nn.utils.clip_grad_norm_(self.get_parameters(name="guide_task_encoder_comp"), self.guide_multitask_cfg.clip_grad_norm)
            self.guide_task_encoder_comp_optimizer.step()
    
    def hindsight_guide_action(
        self,
        batch: ReplayBufferSample,
    ) -> TensorType:
        with torch.no_grad():
            num_envs = self.multitask_cfg.num_envs
            bs = batch.env_obs.shape[0]
            sum_log_prob = torch.zeros(num_envs, bs, 1).to(self.device)
            guide_not_done = batch.not_done[:,0].unsqueeze(0) # [1, bs, 1]
            for i in range(self.guide_step):
                tmp_env_obs = batch.env_obs[:,i].unsqueeze(0).repeat(num_envs, 1, 1).reshape(num_envs * bs, -1)
                tmp_task_obs = torch.tensor([_ for _ in range(num_envs)]).to(self.device).unsqueeze(1).repeat(1, bs).reshape(num_envs * bs)
                if self.should_use_task_encoder:
                    tmp_task_encoding = self.get_task_encoding(
                        env_index=tmp_task_obs,
                        disable_grad=False,
                        modes=["test"],
                    )
                else:
                    tmp_task_encoding = None
                tmp_task_info = self.get_task_info(
                    task_encoding=tmp_task_encoding,
                    compute_grad=False,
                    env_index=tmp_task_obs,
                )
                tmp_mtobs = MTObs(
                    env_obs=tmp_env_obs,
                    task_obs=tmp_task_obs,
                    task_info=tmp_task_info,
                )
                _, _, _, _, dis = self.actor(mtobs=tmp_mtobs, detach_encoder=True, dis_f=True)
                real_action = batch.action[:,i].unsqueeze(0).repeat(num_envs, 1, 1).reshape(num_envs * bs, -1)
                tmp_log_prob = dis.log_prob(real_action).reshape(num_envs, bs, 1)
                if i == 0:
                    sum_log_prob += tmp_log_prob
                else:
                    sum_log_prob += guide_not_done * tmp_log_prob
                guide_not_done *= batch.not_done[:,i].unsqueeze(0)
            hindsight_action = torch.argmax(sum_log_prob, dim=0, keepdim=False)
            return hindsight_action
    
    def update_guide(
        self,
        batch: ReplayBufferSample,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
    ):
        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if self.guide_hindsight:
            hindsighted_guide_action = self.hindsight_guide_action(batch)
            batch.guide_action = hindsighted_guide_action
        
        if self.guide_should_use_task_encoder:
            self.guide_task_encoder_optimizer.zero_grad()
            guide_task_encoding = self.get_guide_task_encoding(
                env_index=batch.task_obs[:,0].squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            guide_task_encoding = None 
            
        task_info = self.get_guide_task_info(
            task_encoding=guide_task_encoding,
            compute_grad=True,
            env_index=batch.task_obs[:,0],
        )
        guide_critic_mask = self.update_guide_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )
        if self.guide_multitask_cfg.use_comp:
            if self.guide_should_use_task_encoder:
                self.guide_task_encoder_comp_optimizer.zero_grad()
                guide_task_encoding_comp = self.get_guide_task_encoding_comp(
                    env_index=batch.task_obs[:,0].squeeze(1),
                    disable_grad=False,
                    modes=["train"],
                )
            else:
                guide_task_encoding_comp = None
            task_info_comp = self.get_guide_task_info(
                task_encoding=guide_task_encoding_comp,
                compute_grad=True,
                env_index=batch.task_obs[:,0],
            )

            guide_critic_comp_mask = self.update_guide_critic_comp(
                batch=batch,
                task_info=task_info,
                task_info_comp=task_info_comp,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        if step % self.guide_actor_update_freq == 0:
            task_info = self.get_guide_task_info(
                task_encoding=guide_task_encoding,
                compute_grad=True,
                env_index=batch.task_obs[:,0],
            )
            self.update_guide_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
                guide_critic_mask=guide_critic_mask,
            )
        if step % self.guide_critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.guide_critic.Q1, self.guide_critic_target.Q1, self.guide_critic_tau
            )
            agent_utils.soft_update_params(
                self.guide_critic.Q2, self.guide_critic_target.Q2, self.guide_critic_tau
            )
            agent_utils.soft_update_params(
                self.guide_critic.encoder, self.guide_critic_target.encoder, self.guide_encoder_tau
            )
            if self.guide_multitask_cfg.use_comp:
                agent_utils.soft_update_params(
                    self.guide_critic_comp.Q1, self.guide_critic_comp_target.Q1, self.guide_critic_tau
                )
                agent_utils.soft_update_params(
                    self.guide_critic_comp.Q2, self.guide_critic_comp_target.Q2, self.guide_critic_tau
                )
                agent_utils.soft_update_params(
                    self.guide_critic_comp.encoder, self.guide_critic_comp_target.encoder, self.guide_encoder_tau
                )

        if self.guide_should_use_task_encoder:
            self.update_guide_task_encoder(
                batch=batch,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        
        # PaCo Trick
        if step > self.guide_multitask_cfg.mask_loss_step:
            self._guide_reset_task_while_extreme(guide_critic_mask, ~self.no_guide.to(torch.bool), logger, step)
            if self.guide_multitask_cfg.use_comp:
                self._guide_reset_task_while_extreme(guide_critic_comp_mask, ~self.no_guide.to(torch.bool), logger, step, comp_f=True)
        
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
        self.train_step = step
        buffer_index_to_sample = super().update(
            replay_buffer=replay_buffer,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=kwargs_to_compute_gradient,
            buffer_index_to_sample=buffer_index_to_sample,
        )
        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if step % self.guide_multitask_cfg.guide_update_step == 0:
            batch = replay_buffer.guide_sample(scale=self.guide_sample_scale)

            if self.guide_loss_reduction == "alpha_weight":
                softmax_temp = F.softmax(-self.log_alpha.detach())
                self.guide_loss_alpha_weight = softmax_temp * self.num_envs

            self.update_guide(
                batch=batch,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=kwargs_to_compute_gradient,
            )

            return batch.buffer_index
        return buffer_index_to_sample
    
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
        elif name == "guide_actor":
            return list(self.guide_actor.model.parameters())
        elif name in ["guide_log_alpha", "guide_alpha"]:
            return [self.guide_log_alpha]
        elif name == "guide_encoder":
            return list(self.guide_critic.encoder.parameters())
        else:
            return list(self._components[name].parameters())
    
    def update_no_guide(self, success=None):
        guide_prob_mode = self.guide_multitask_cfg.guide_prob_mode
        if guide_prob_mode == "alpha":
            log_alpha = self.log_alpha.detach() 
            mean_log_alpha = log_alpha.mean()
            self.no_guide_prob = (log_alpha > mean_log_alpha).long()
        elif guide_prob_mode == "success":
            self.no_guide_prob = (torch.tensor(success, device=self.no_guide_prob.device) > 0.8).long()
        else:
            self.no_guide_prob = torch.zeros_like(self.no_guide_prob)
        print("no_guide:", self.no_guide_prob)
        self.update_guide_sample_sacle(self.no_guide_prob.float().cpu().numpy())

    def reset_at_begin(self, env_idx: int, step: int):
        if self.guide_multitask_cfg.use_no_guide:
            self.no_guide[env_idx] = self.no_guide_prob[env_idx].long()
        
        # reset guide_policy
        self.guide_policy[env_idx] = -1
        self.guide_step_already[env_idx] = 0
    
    def _guide_reset_task_while_extreme(self, task_mask, valid_mask, logger, step, comp_f=False):
        """
        PaCo Tricks
        """
        if self.guide_should_use_task_encoder and self.guide_multitask_cfg.actor_cfg.paco_should_use:
            extreme_task_mask = ~task_mask
            if comp_f:
                self.guide_task_encoder_comp.reset_weight_by_task_mask(extreme_task_mask, valid_mask)
            else:
                self.guide_task_encoder.reset_weight_by_task_mask(extreme_task_mask, valid_mask)
    
    def update_guide_sample_sacle(self, no_guide_prob: np.ndarray):
        guide_prob = 1 - no_guide_prob
        if sum(guide_prob) == 0:
            guide_prob = np.ones_like(guide_prob)
        self.guide_sample_scale = guide_prob / sum(guide_prob)

        