# -*- coding: utf-8 -*-
import os
import json
import torch
import zipfile
from tatk.util.file_util import cached_path
from tatk.policy.policy import Policy
from tatk.policy.vhus.util import capital, padding
from tatk.task.multiwoz.goal_generator import GoalGenerator
from tatk.policy.vhus.multiwoz.usermanager import UserDataManager
from tatk.policy.vhus.usermodule import VHUS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "vhus_simulator_multiwoz.zip")

class UserPolicyVHUS(Policy):

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

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for VHUS Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_simulator.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        self.load(config['load'])

    def init_session(self):
        self.time_step = -1
        self.topic = 'NONE'
        self.goal = self.goal_gen.get_user_goal()
        self.goal_input = torch.LongTensor(self.manager.get_goal_id(self.manager.usrgoal2seq(self.goal)))
        self.goal_len_input = torch.LongTensor([len(self.goal_input)]).squeeze()
        self.sys_da_id_stack = []  # to save sys da history

    def predict(self, state):
        """Predict an user act based on state and preorder system action.

        Args:
            state (tuple):
                Dialog state.
        Returns:
            usr_action (tuple):
                User act.
            session_over (boolean):
                True to terminate session, otherwise session continues.
        """
        sys_action = state
		
        sys_seq_turn = self.manager.sysda2seq(self.manager.ref_data2stand(sys_action), self.goal)
        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor([max(len(sen), 1) for sen in self.sys_da_id_stack])
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(padding(self.sys_da_id_stack, max_sen_len))
        usr_a, terminal = self.user.select_action(self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self.manager.usrseq2da(self.manager.id2sentence(usr_a), self.goal)

        return capital(usr_action), terminal

    def load(self, filename):
        user_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_simulator.mdl')
        if os.path.exists(user_mdl):
            self.user.load_state_dict(torch.load(user_mdl))
            print('<<user simulator>> loaded checkpoint from file: {}'.format(user_mdl))

    def get_goal(self):
        return self.goal
