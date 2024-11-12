# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from mtrl.utils.types import ConfigType
from mtrl.mtenv.envs.wrappers import SeedWrapper, ScaleRewardWrapper, MaxStepSuccessWrapper

import metaworld
from gym import Env, Wrapper

from mtrl.mtenv.envs.gym_extensions import register_custom_envs
import gym

EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int
EnvIdToTaskMapType = Dict[str, metaworld.Task]


def get_list_of_func_to_make_envs(
    config: ConfigType,
    benchmark_name: str,
    num_copies_per_env: int = 1,
) -> List[Any]:
    """Return a list of functions to construct the MetaWorld environments
    and a mapping of environment ids to tasks.

    Args:
        config (ConfigType): config of experiment
        benchmark_name (str): name of the `benchmark`.
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.

    Raises:
        ValueError: if `benchmark_name` is not MT4.

    Returns:
        List[Any]: a list of functions to construct the MetaWorld environments.

    """
    # Modified gravity
    if benchmark_name == "ant_gravity.MT4":
        env_names = ["Ant-v2",
                     "AntGravityMars-v0", "AntGravityHalf-v0", "AntGravityOneAndHalf-v0"]
    elif benchmark_name == "hopper_gravity.MT5":
        env_names = ["Hopper-v2",
                     "HopperGravityHalf-v0", "HopperGravityThreeQuarters-v0", "HopperGravityOneAndHalf-v0", "HopperGravityOneAndQuarter-v0"]
    elif benchmark_name == "walker2d_gravity.MT5":
        env_names = ["Walker2d-v2",
                     "Walker2dGravityHalf-v0", "Walker2dGravityThreeQuarters-v0", "Walker2dGravityOneAndHalf-v0", "Walker2dGravityOneAndQuarter-v0"]
    elif benchmark_name == "halfcheetah_gravity.MT5":
        env_names = ["HalfCheetah-v2",
                     "HalfCheetahGravityHalf-v0", "HalfCheetahGravityThreeQuarters-v0", "HalfCheetahGravityOneAndHalf-v0", "HalfCheetahGravityOneAndQuarter-v0"]
    elif benchmark_name == "humanoid_gravity.MT5":
        env_names = ["Humanoid-v2",
                     "HumanoidGravityHalf-v0", "HumanoidGravityThreeQuarters-v0", "HumanoidGravityOneAndHalf-v0", "HumanoidGravityOneAndQuarter-v0"]
    # Modified body parts
    elif benchmark_name == "hopper_body.MT8":
        env_names = ["HopperBigTorso-v0", "HopperSmallTorso-v0",
                     "HopperBigThigh-v0", "HopperSmallThigh-v0",
                     "HopperBigLeg-v0", "HopperSmallLeg-v0",
                     "HopperBigFoot-v0", "HopperSmallFoot-v0"]
    elif benchmark_name == "walker2d_body.MT8":
        env_names = ["Walker2dBigTorso-v0", "Walker2dSmallTorso-v0",
                     "Walker2dBigThigh-v0", "Walker2dSmallThigh-v0",
                     "Walker2dBigLeg-v0", "Walker2dSmallLeg-v0",
                     "Walker2dBigFoot-v0", "Walker2dSmallFoot-v0"]
    elif benchmark_name == "halfcheetah_body.MT8":
        env_names = ["HalfCheetahBigTorso-v0", "HalfCheetahSmallTorso-v0",
                     "HalfCheetahBigThigh-v0", "HalfCheetahSmallThigh-v0",
                     "HalfCheetahBigLeg-v0", "HalfCheetahSmallLeg-v0",
                     "HalfCheetahBigFoot-v0", "HalfCheetahSmallFoot-v0"]
    elif benchmark_name == "humanoid_body.MT14":
        env_names = ["HumanoidBigTorso-v0", "HumanoidSmallTorso-v0",
                     "HumanoidBigThigh-v0", "HumanoidSmallThigh-v0",
                     "HumanoidBigLeg-v0", "HumanoidSmallLeg-v0",
                     "HumanoidBigFoot-v0", "HumanoidSmallFoot-v0",
                     "HumanoidBigHead-v0", "HumanoidSmallHead-v0",
                     "HumanoidBigArm-v0", "HumanoidSmallArm-v0",
                     "HumanoidBigHand-v0", "HumanoidSmallHand-v0"]
    # No other benchmarks
    else:
        raise ValueError(f"benchmark_name={benchmark_name} is not valid in walker2d.")

    env_id_list = env_names

    def get_func_to_make_envs(env_id: str):

        def _make_env():
            env_dict = gym.envs.registration.registry.env_specs.copy()
            if "HopperBigTorso-v0" not in env_dict:
                register_custom_envs()
            del env_dict
            for name in env_names:
                if name == env_id:
                    env = gym.make(name)
                    env = SeedWrapper(env)
                    if 'wrappers' in config.env:
                        if 'scale_reward_wrapper' in config.env.wrappers:
                            env = ScaleRewardWrapper(env, reward_scale=config.env.wrappers.scale_reward_wrapper.reward_scale)
                        if 'max_step_success_wrapper' in config.env.wrappers:
                            env = MaxStepSuccessWrapper(env, max_step=config.env.wrappers.max_step_success_wrapper.max_step)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        # without test
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    funcs_to_make_envs = [get_func_to_make_envs(env_id) for env_id in env_id_list]

    return funcs_to_make_envs, env_names

