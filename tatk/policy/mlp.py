# -*- coding: utf-8 -*-
import torch
import numpy as np
import logging
from tatk.policy.policy import Policy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(Policy):
    
    def __init__(self, is_train=False):
        self.is_train = is_train
        
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
