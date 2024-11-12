# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy

from mtrl.mtenv.envs.registration import register

default_kwargs = {
    "benchmark": None,
    "benchmark_name": "MT10",
    "env_id_to_task_map": None,
    "random_goal": False,
    "num_copies_per_env": 1,
    "initial_task_state": 1,
}

for benchmark_name in [("MT10"), ("MT50")]:
    kwargs = deepcopy(default_kwargs)
    kwargs["benchmark_name"] = benchmark_name
    register(
        id=f"MT-MetaWorld-{benchmark_name}-v0",
        entry_point="mtrl.mtenv.envs.metaworld.env:build",
        kwargs=kwargs,
        test_kwargs={
            # "valid_env_kwargs": [],
            # "invalid_env_kwargs": [],
        },
    )

kwargs = {
    "benchmark": None,
    "benchmark_name": "MT1",
    "env_id_to_task_map": None,
    "random_goal": False,
    "task_name": "pick-place-v2",
    "num_copies_per_env": 1,
    "initial_task_state": 0,
}
register(
    id=f'MT-MetaWorld-{kwargs["benchmark_name"]}-v0',
    entry_point="mtrl.mtenv.envs.metaworld.env:build",
    kwargs=kwargs,
    test_kwargs={
        # "valid_env_kwargs": [],
        # "invalid_env_kwargs": [],
    },
)
