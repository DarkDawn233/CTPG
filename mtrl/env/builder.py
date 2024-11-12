# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, Optional, Tuple

from gym.vector.async_vector_env import AsyncVectorEnv

from mtrl.env.vec_env import MetaWorldVecEnv, GymExtensionsVecEnv
from mtrl.utils.types import ConfigType

def build_metaworld_vec_env(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    from mtrl.mtenv.envs.metaworld.env import (
        get_list_of_func_to_make_envs as get_list_of_func_to_make_metaworld_envs,
    )
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", ""))
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name,
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "random_goal": config.env.random_goal,
    }

    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        config=config,
        **make_kwargs
    )
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    env.seed(config.setup.seed)
    return env, env_id_to_task_map


def build_gym_extensions_vec_env(
    config: ConfigType,
    mode: str,
) -> AsyncVectorEnv:
    from mtrl.mtenv.envs.gym_extensions.env import (
        get_list_of_func_to_make_envs as get_list_of_func_to_make_gym_extensions_envs,
    )
    benchmark_name = config.env.benchmark_name
    num_tasks = int(benchmark_name.split(".MT")[-1])
    make_kwargs = {
        "benchmark_name": benchmark_name,
        "num_copies_per_env": 1,
    }

    funcs_to_make_envs, task_list = get_list_of_func_to_make_gym_extensions_envs(
        config=config, **make_kwargs
    )
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = GymExtensionsVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    env.seed(config.setup.seed)
    return env, task_list
