# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from typing import List, Union
from mtrl.utils.types import TensorType


@dataclass
class ReplayBufferSample:
    env_obs: TensorType = None
    action: TensorType = None
    reward: TensorType = None
    next_env_obs: TensorType = None
    not_done: TensorType = None
    task_obs: TensorType = None
    guide_action: TensorType = None
    guide_begin: TensorType = None
    buffer_index: Union[np.ndarray, List[np.ndarray]] = None

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self, env_obs_shape, task_obs_shape, action_shape, capacity, batch_size, task_num, device,
        agent_cfg, experiment_cfg
    ):
        assert capacity % task_num == 0, f"Capacity is not dividable by task_num"
        self.task_num = task_num
        self.task_obs_should_be = torch.tensor([_ for _ in range(task_num)]).reshape(-1, 1)
        self.single_capacity = capacity // task_num
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.agent_cfg = agent_cfg
        self.experiment_cfg = experiment_cfg

        # the proprioceptive env_obs is stored as float32, pixels env_obs as uint8
        env_obs_dtype = np.float32 if len(env_obs_shape) == 1 else np.uint8
        task_obs_dtype = np.int64

        self.env_obses = np.empty((self.single_capacity, task_num, *env_obs_shape), dtype=env_obs_dtype)
        self.next_env_obses = np.empty((self.single_capacity, task_num, *env_obs_shape), dtype=env_obs_dtype)
        self.actions = np.empty((self.single_capacity, task_num, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.single_capacity, task_num, 1), dtype=np.float32)
        self.not_dones = np.empty((self.single_capacity, task_num, 1), dtype=np.float32)
        self.task_obs = np.empty((self.single_capacity, task_num, *task_obs_shape), dtype=task_obs_dtype)

        self.save_guide_action = self.experiment_cfg.use_guide if 'use_guide' in self.experiment_cfg else False
        if self.save_guide_action:
            self.guide_actions = np.empty((self.single_capacity, task_num, 1), dtype=np.int32)
            self.guide_indexs = [[] for _ in range(task_num)]
            self.guide_step = self.experiment_cfg.guide_step

        self.idx = 0
        self.last_save = 0
        self.full = False

    def is_empty(self):
        return not self.full and self.idx == 0

    def add(self, env_obs, action, reward, next_env_obs, not_done, task_obs,
            guide_action=None, guide_begin=None):
        assert (task_obs == self.task_obs_should_be).all()
        np.copyto(self.env_obses[self.idx], env_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_env_obses[self.idx], next_env_obs)
        np.copyto(self.not_dones[self.idx], not_done)
        np.copyto(self.task_obs[self.idx], task_obs)

        if self.save_guide_action:
            np.copyto(self.guide_actions[self.idx], guide_action)
            for i in range(self.task_num):
                if len(self.guide_indexs[i]) != 0 and self.guide_indexs[i][0] == self.idx:
                    self.guide_indexs[i].pop(0)
                if guide_begin[i][0]:
                    self.guide_indexs[i].append(self.idx)

        self.idx = (self.idx + 1) % self.single_capacity
        self.full = self.full or self.idx == 0

    def sample(self, index=None, scale=None) -> ReplayBufferSample:
        if scale is not None:
            assert len(scale) == self.task_num and (sum(scale) - 1) < 1e-4
        else:
            scale = [None for _ in range(self.task_num)]
        env_obses, actions, rewards, next_env_obses, not_dones, env_indices = [], [], [], [], [], []

        idxs = []
        for task_i, scale_i in enumerate(scale):
            if index is None:
                if scale_i is None:
                    idx = np.random.randint(
                        0, self.single_capacity if self.full else self.idx, size=self.batch_size // self.task_num
                    )
                else:
                    idx = np.random.randint(
                        0, self.single_capacity if self.full else self.idx, size=round(self.batch_size * scale_i)
                    )
            else:
                if isinstance(index, list):
                    idx = index[task_i]
                else:
                    idx = index
            env_obses.append(torch.as_tensor(self.env_obses[idx, task_i], device=self.device).float())
            actions.append(torch.as_tensor(self.actions[idx, task_i], device=self.device))
            rewards.append(torch.as_tensor(self.rewards[idx, task_i], device=self.device))
            next_env_obses.append(torch.as_tensor(self.next_env_obses[idx, task_i], device=self.device).float())
            not_dones.append(torch.as_tensor(self.not_dones[idx, task_i], device=self.device))
            env_indices.append(torch.as_tensor(self.task_obs[idx, task_i], device=self.device))
            idxs.append(idx)

        env_obses = torch.concat(env_obses, dim=0)
        actions = torch.concat(actions, dim=0)
        rewards = torch.concat(rewards, dim=0)
        next_env_obses = torch.concat(next_env_obses, dim=0)
        not_dones = torch.concat(not_dones, dim=0)
        env_indices = torch.concat(env_indices, dim=0)

        return ReplayBufferSample(
            env_obses, actions, rewards, next_env_obses, not_dones, env_indices,
            guide_action=None, buffer_index=idxs
        )

    def guide_sample(self, index=None, scale=None) -> ReplayBufferSample:
        if scale is not None:
            assert len(scale) == self.task_num and (sum(scale) - 1) < 1e-4
        else:
            scale = [None for _ in range(self.task_num)]
        env_obses, actions, rewards, next_env_obses, not_dones, env_indices = [], [], [], [], [], []
        guide_actions = [] if self.save_guide_action else None
        idxs = []
        for task_i, scale_i in enumerate(scale):
            if index is None:
                if scale_i is None:
                    replace = False if len(self.guide_indexs[task_i][:-1]) >= self.batch_size // self.task_num else True
                    idx = np.random.choice(self.guide_indexs[task_i][:-1], size=self.batch_size // self.task_num, replace=replace)
                else:
                    replace = False if len(self.guide_indexs[task_i][:-1]) >= round(self.batch_size * scale_i) else True
                    idx = np.random.choice(self.guide_indexs[task_i][:-1], size=round(self.batch_size * scale_i), replace=replace)
            else:
                if isinstance(index, list):
                    idx = index[task_i]
                else:
                    idx = index
            multi_env_obses, multi_actions, multi_rewards, multi_next_env_obses, multi_not_dones, multi_env_indices = [], [], [], [], [], []
            for i in range(self.guide_step):
                idx_now = (idx+i) % self.single_capacity if self.full else np.minimum(idx+i, self.idx-1)
                multi_env_obses.append(torch.as_tensor(self.env_obses[idx_now, task_i], device=self.device).float())
                multi_actions.append(torch.as_tensor(self.actions[idx_now, task_i], device=self.device))
                multi_rewards.append(torch.as_tensor(self.rewards[idx_now, task_i], device=self.device))
                multi_next_env_obses.append(torch.as_tensor(self.next_env_obses[idx_now, task_i], device=self.device).float())
                multi_not_dones.append(torch.as_tensor(self.not_dones[idx_now, task_i], device=self.device))
                multi_env_indices.append(torch.as_tensor(self.task_obs[idx_now, task_i], device=self.device))
            env_obses.append(torch.stack(multi_env_obses, dim=1))
            actions.append(torch.stack(multi_actions, dim=1))
            rewards.append(torch.stack(multi_rewards, dim=1))
            next_env_obses.append(torch.stack(multi_next_env_obses, dim=1))
            not_dones.append(torch.stack(multi_not_dones, dim=1))
            env_indices.append(torch.stack(multi_env_indices, dim=1))

            if self.save_guide_action:
                guide_actions.append(torch.as_tensor(self.guide_actions[idx, task_i], device=self.device))
            idxs.append(idx)

        env_obses = torch.concat(env_obses, dim=0)
        actions = torch.concat(actions, dim=0)
        rewards = torch.concat(rewards, dim=0)
        next_env_obses = torch.concat(next_env_obses, dim=0)
        not_dones = torch.concat(not_dones, dim=0)
        env_indices = torch.concat(env_indices, dim=0)
        if self.save_guide_action:
            guide_actions = torch.concat(guide_actions, dim=0)

        return ReplayBufferSample(
            env_obses, actions, rewards, next_env_obses, not_dones, env_indices,
            guide_action=guide_actions, buffer_index=idxs
        )

    def _sample_a_replay_buffer(self, num_samples):
        """This method returns a new replay buffer which contains samples from the original replay buffer.
        For now, this is meant to be used only when saving a replay buffer.
        """
        indices = np.random.choice(
            self.single_capacity if self.full else self.idx, num_samples, replace=False
        )
        new_replay_buffer = ReplayBuffer(
            env_obs_shape=self.env_obses.shape[1:],
            action_shape=self.actions.shape[1:],
            capacity=num_samples,
            batch_size=self.batch_size,
            device=self.device,
            extra_cfg=self.extra_cfg
        )
        new_replay_buffer.env_obses = self.env_obses[indices]
        new_replay_buffer.next_env_obses = self.next_env_obses[indices]
        new_replay_buffer.actions = self.actions[indices]
        new_replay_buffer.rewards = self.rewards[indices]
        new_replay_buffer.not_dones = self.not_dones[indices]
        new_replay_buffer.task_obs = self.task_obs[indices]
        if self.save_guide_action:
            new_replay_buffer.guide_actions = self.guide_actions[indices]
        return new_replay_buffer

    def delete_from_filesystem(self, dir_to_delete_from: str):
        for filename in os.listdir(dir_to_delete_from):
            file_path = os.path.join(dir_to_delete_from, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"Deleted files from: {dir_to_delete_from}")

    def save(self, save_dir, size_per_chunk: int, num_samples_to_save: int):
        num_samples_to_save //= self.task_num
        if self.idx == self.last_save:
            return
        if num_samples_to_save == -1:
            self._save_all(
                save_dir=save_dir,
                size_per_chunk=size_per_chunk,
            )
        else:
            if num_samples_to_save > self.idx:
                num_samples_to_save = self.idx
                replay_buffer_to_save = self
            else:
                replay_buffer_to_save = self._sample_a_replay_buffer(
                    num_samples=num_samples_to_save
                )
                replay_buffer_to_save.idx = num_samples_to_save
                replay_buffer_to_save.last_save = 0
            backup_dir_path = Path(f"{save_dir}_bk")
            if not backup_dir_path.exists():
                backup_dir_path.mkdir()
            replay_buffer_to_save._save_all(
                save_dir=str(backup_dir_path),
                size_per_chunk=size_per_chunk,
            )
            replay_buffer_to_save.delete_from_filesystem(dir_to_delete_from=save_dir)
            backup_dir_path.rename(save_dir)
        self.last_save = self.idx

    def _save_all(self, save_dir, size_per_chunk: int):
        if self.idx == self.last_save:
            return
        if self.last_save == self.single_capacity:
            self.last_save = 0
        if self.idx > self.last_save:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        else:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.single_capacity,
                size_per_chunk=size_per_chunk,
            )
            self._save_payload(
                save_dir=save_dir,
                start_idx=0,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        self.last_save = self.idx

    def _save_payload(
        self, save_dir: str, start_idx: int, end_idx: int, size_per_chunk: int
    ):
        while True:
            if size_per_chunk > 0:
                current_end_idx = min(start_idx + size_per_chunk, end_idx)
            else:
                current_end_idx = end_idx
            self._save_payload_chunk(
                save_dir=save_dir, start_idx=start_idx, end_idx=current_end_idx
            )
            if current_end_idx == end_idx:
                break
            start_idx = current_end_idx

    def _save_payload_chunk(self, save_dir: str, start_idx: int, end_idx: int):
        path = os.path.join(save_dir, f"{start_idx}_{end_idx-1}.pt")
        payload = [
            self.env_obses[start_idx:end_idx],
            self.next_env_obses[start_idx:end_idx],
            self.actions[start_idx:end_idx],
            self.rewards[start_idx:end_idx],
            self.not_dones[start_idx:end_idx],
            self.task_obs[start_idx:end_idx],
        ]
        if self.save_guide_action:
            payload.append(self.guide_actions[start_idx:end_idx])
        print(f"Saving replay buffer at {path}")
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        start = 0
        for chunk in chunks:
            path = os.path.join(save_dir, chunk)
            try:
                payload = torch.load(path)
                end = start + payload[0].shape[0]
                if end > self.single_capacity:
                    # this condition is added for resuming some very old experiments.
                    # This condition should not be needed with the new experiments
                    # and should be removed going forward.
                    select_till_index = payload[0].shape[0] - (end - self.single_capacity)
                    end = start + select_till_index
                else:
                    select_till_index = payload[0].shape[0]
                self.env_obses[start:end] = payload[0][:select_till_index]
                self.next_env_obses[start:end] = payload[1][:select_till_index]
                self.actions[start:end] = payload[2][:select_till_index]
                self.rewards[start:end] = payload[3][:select_till_index]
                self.not_dones[start:end] = payload[4][:select_till_index]
                self.task_obs[start:end] = payload[5][:select_till_index]
                payload_i = 6
                if self.save_guide_action:
                    self.guide_actions[start:end] = payload[payload_i][:select_till_index]
                    payload_i += 1
                self.idx = end - 1
                start = end
                print(f"Loaded replay buffer from path: {path})")
            except EOFError as e:
                print(
                    f"Skipping loading replay buffer from path: {path} due to error: {e}"
                )
        self.last_save = self.idx

    def reset(self):
        self.idx = 0
