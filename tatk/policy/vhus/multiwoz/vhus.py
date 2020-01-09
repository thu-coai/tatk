# -*- coding: utf-8 -*-
import os
import json
import torch
from tatk.policy.vhus.util import capital
from tatk.task.multiwoz.goal_generator import GoalGenerator
from tatk.policy.vhus.multiwoz.usermanager import UserDataManager
from tatk.policy.vhus.usermodule import VHUS
from tatk.policy.vhus.vhus import UserPolicyVHUSAbstract

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "vhus_simulator_multiwoz.zip")

class UserPolicyVHUS(UserPolicyVHUSAbstract):

    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/vhus_simulator_multiwoz.zip'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            config = json.load(f)
        manager = UserDataManager()
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size).to(device=DEVICE)
        self.goal_gen = GoalGenerator()
        self.manager = manager
        self.user.eval()

        self.load(archive_file, model_file, config['load'])

    def predict(self, state):
        usr_action = super().predict(state)
        return capital(usr_action)
