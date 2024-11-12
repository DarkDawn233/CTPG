import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math

import numpy as np
import gym
import random
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

def GravityEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class GravityEnv(class_type, utils.EzPickle):
        """
        Allows the gravity to be changed by the
        """
        def __init__(
                self,
                model_path,
                gravity=-9.81,
                *args,
                **kwargs):
            
            assert isinstance(self, mujoco_env.MujocoEnv)

            tree = ET.parse(model_path)

            option = tree.find(".//option")
            new_gravity = "0 0 " + str(gravity)
            option.attrib["gravity"] = new_gravity
            # print("new_gravity:", option.attrib["gravity"])

            # create new xml
            _, file_path = tempfile.mkstemp(suffix=".xml", text=True)
            tree.write(file_path)

            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

    return GravityEnv
