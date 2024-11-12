# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from mtrl.utils.types import ConfigType
from mtrl.mtenv.envs.wrappers import SeedWrapper, ScaleRewardWrapper, MaxStepWrapper

import metaworld
from gym import Env

EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int
EnvIdToTaskMapType = Dict[str, metaworld.Task]


def get_list_of_func_to_make_envs(
    config: ConfigType,
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    random_goal: bool = False,
    task_name: str = "pick-place-v2",
    num_copies_per_env: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return a list of functions to construct the MetaWorld environments
    and a mapping of environment ids to tasks.

    Args:
        benchmark (Optional[metaworld.Benchmark]): `benchmark` to create
            tasks from.
        benchmark_name (str): name of the `benchmark`. This is used only
            when the `benchmark` is None.
        env_id_to_task_map (Optional[EnvIdToTaskMapType]): In MetaWorld,
            each environment can be associated with multiple tasks. This
            dict persists the mapping between environment ids and tasks.
        task_name (str, optional): In case of MT1, only . Defaults to
            "pick-place-v2".
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.

    Raises:
        ValueError: if `benchmark` is None and `benchmark_name` is not
            MT1, MT10, or MT50.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple of two elements. The
        first element is a list of functions to construct the MetaWorld
        environments and the second is a mapping of environment ids
        to tasks.

    """
    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def get_func_to_make_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    env._freeze_rand_vec = not random_goal
                    env = SeedWrapper(env)
                    if 'wrappers' in config.env:
                        if 'scale_reward_wrapper' in config.env.wrappers:
                            env = ScaleRewardWrapper(env, reward_scale=config.env.wrappers.scale_reward_wrapper.reward_scale)
                        if 'max_step_wrapper' in config.env.wrappers:
                            env = MaxStepWrapper(env, max_step=config.env.wrappers.max_step_wrapper.max_step)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    funcs_to_make_envs = [get_func_to_make_envs(env_id) for env_id in env_id_list]

    return funcs_to_make_envs, env_id_to_task_map

