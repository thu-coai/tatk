# -*- coding: utf-8 -*-
import torch
import numpy as np
import logging
import os
from tatk.policy.policy import Policy
from tatk.policy.rlmodule import MultiDiscretePolicy
from tatk.policy.multiwoz.vector_multiwoz import MultiWozVector
from tatk.policy.camrest.vector_camrest import CamrestVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(Policy):
    
    def __init__(self, cfg, is_train=False, dataset='multiwoz'):
        self.is_train = is_train
        self.policy = MultiDiscretePolicy(cfg).to(device=DEVICE)
        
        if dataset == 'multiwoz':
            voc_file = os.path.join(
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
        elif dataset == 'camrest':
            voc_file = os.path.join(
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data/camrest/sys_da_voc.txt')
            voc_opp_file = os.path.join(
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'data/camrest/usr_da_voc.txt')
            self.vector = CamrestVector(voc_file, voc_opp_file)
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
        a = self.policy.select_action(s_vec.to(device=DEVICE)).cpu()
        action = self.vector.action_devectorize(a.numpy())
        
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass
