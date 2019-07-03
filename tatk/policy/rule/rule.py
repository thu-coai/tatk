# -*- coding: utf-8 -*-
import torch
import numpy as np
import logging
import os
from tatk.policy.policy import Policy
from tatk.policy.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot
from tatk.policy.multiwoz.policy_agenda_multiwoz import UserPolicyAgendaMultiWoz
from tatk.policy.multiwoz.vector_multiwoz import MultiWozVector
from tatk.policy.camrest.rule_based_camrest_bot import RuleBasedCamrestBot
from tatk.policy.camrest.policy_agenda_camrest import UserPolicyAgendaCamrest
from tatk.policy.camrest.vector_camrest import CamrestVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Rule(Policy):
    
    def __init__(self, is_train=False, dataset='multiwoz', character='sys'):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.is_train = is_train
        if dataset == 'multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            if character == 'sys':
                self.vector = MultiWozVector(voc_file, voc_opp_file)
                self.policy = RuleBasedMultiwozBot()
            elif character == 'usr':
                self.vector = MultiWozVector(voc_opp_file, voc_file)
                self.policy = UserPolicyAgendaMultiWoz()
            else:
                raise NotImplementedError('unknown character {}'.format(character))
        elif dataset == 'camrest':
            voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
            if character == 'sys':
                self.vector = CamrestVector(voc_file, voc_opp_file)
                self.policy = RuleBasedCamrestBot()
            elif character == 'usr':
                self.vector = CamrestVector(voc_opp_file, voc_file)
                self.policy = UserPolicyAgendaCamrest()
            else:
                raise NotImplementedError('unknown character {}'.format(character))
        else:
            raise NotImplementedError('unknown dataset {}'.format(dataset))
        
    def predict(self, state, sess=None):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.predict(s_vec.to(device=DEVICE)).cpu()
        action = self.vector.action_devectorize(a.numpy())
        
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass
