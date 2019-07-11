# -*- coding: utf-8 -*-
import os
import json
import torch

from tatk.policy.policy import Policy
from tatk.policy.vhus.util import padding
from tatk.task.camrest.goal_generator import GoalGenerator
from tatk.policy.vhus.camrest.usermanager import UserDataManager
from tatk.policy.vhus.usermodule import VHUS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class UserPolicyVHUS(Policy):
    
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            config = json.load(f)
        manager = UserDataManager()
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size).to(device=DEVICE)
        self.goal_gen = GoalGenerator()
        self.manager = manager
        self.user.eval()
        
        self.load(config['load'])
        
    def init_session(self):
        self.time_step = -1
        self.topic = 'NONE'
        self.goal = self.goal_gen.get_user_goal()
        self.goal_input = torch.LongTensor(self.manager.get_goal_id(self.manager.usrgoal2seq(self.goal)))
        self.goal_len_input = torch.LongTensor([len(self.goal_input)]).squeeze()
        self.sys_da_id_stack = [] # to save sys da history

    def predict(self, state):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
        Returns:
            usr_action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
        """
        sys_action = state['system_action']
        sys_seq_turn = self.manager.sysda2seq(self.manager.ref_data2stand(sys_action), self.goal)
        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor([max(len(sen), 1) for sen in self.sys_da_id_stack])
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(padding(self.sys_da_id_stack, max_sen_len))
        usr_a, terminal = self.user.select_action(self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self.manager.usrseq2da(self.manager.id2sentence(usr_a), self.goal)
        
        return usr_action, terminal
    
    def load(self, filename):
        user_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_simulator.mdl')
        if os.path.exists(user_mdl):
            self.user.load_state_dict(torch.load(user_mdl))
            print('<<user simulator>> loaded checkpoint from file: {}'.format(user_mdl))