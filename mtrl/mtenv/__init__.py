# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
__version__ = "1.0"

from mtrl.mtenv.core import MTEnv  # noqa: F401
from mtrl.mtenv.envs.registration import make  # noqa: F401

__all__ = ["MTEnv", "make"]
