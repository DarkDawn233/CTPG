"""Class to interface with an Experiment"""

from typing import Dict

import hydra
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import multitask
from mtrl.utils.types import ConfigType


class Experiment(multitask.Experiment):
    """Experiment Class"""

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))

        return eval_modes_to_env_ids

    def build_envs(self):
        envs = {}
        mode = "train"
        envs[mode], task_list = env_builder.build_gym_extensions_vec_env(
            config=self.config, mode=mode
        )
        mode = "eval"
        envs[mode], task_list = env_builder.build_gym_extensions_vec_env(
            config=self.config, mode=mode
        )

        max_episode_steps = self.config.env.wrappers.max_step_wrapper.max_step \
                                if "max_step_wrapper" in self.config.env.wrappers else 1000
        # hardcoding the steps as different environments return different
        # values for max_path_length. Walker2D uses 1000 as the max length.
        metadata = self.get_env_metadata(
            env=envs["train"],
            max_episode_steps=max_episode_steps,
            ordered_task_list=task_list,
        )
        return envs, metadata

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int) -> Dict: # TODO
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/step", step, step)

        agent = self.agent
        num_eval_episodes = self.config.experiment.num_eval_episodes

        eval_episode_reward, eval_success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0.0]
        ]

        eval_env_step = np.zeros(vec_env.num_envs)

        for i in range(num_eval_episodes):
            episode_step = 0
            episode_reward, mask, ep_done, success = [
                np.full(shape=vec_env.num_envs, fill_value=fill_value)
                for fill_value in [0.0, 1.0, False, 0.0]
            ]
            multitask_obs = vec_env.reset()
            while not ep_done.all():
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        multitask_obs=multitask_obs, modes=["eval"]
                    )
                multitask_obs, reward, done, info = vec_env.step(action)
                for env_i in range(vec_env.num_envs):
                    if done[env_i] and not ep_done[env_i]:
                        eval_env_step[env_i] += info[env_i]["env_step"]
                ep_done = np.logical_or(done, ep_done)
                episode_reward += reward * mask
                success += np.asarray([x["success"] for x in info]) * mask
                mask = mask * (1 - ep_done.astype("int"))
                episode_step += 1
            success = (success > 0).astype("float")
            eval_episode_reward += episode_reward
            eval_success += success

        eval_episode_reward /= num_eval_episodes
        eval_success /= num_eval_episodes
        eval_env_step /= num_eval_episodes

        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode_reward", eval_episode_reward.mean(), step)
            self.logger.log(f"{mode}/success", eval_success.mean(), step)
            for _env_id in self.eval_modes_to_env_ids[mode]:
                self.logger.log(f"{mode}/episode_reward_{_env_id}", 
                                eval_episode_reward[_env_id], step)
                self.logger.log(f"{mode}/success_{_env_id}",
                                eval_success[_env_id], step)
                self.logger.log(f"{mode}/env_step_{_env_id}",
                                eval_env_step[_env_id], step)
        self.logger.dump(step)

        return {'success': eval_success, 'reward': eval_episode_reward}