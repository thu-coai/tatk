# -*- coding: utf-8 -*-
import torch
import os
import json
from tatk.policy.policy import Policy
from tatk.policy.rlmodule import MultiDiscretePolicy
from tatk.policy.vector.vector_camrest import CamrestVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(Policy):
    
    def __init__(self, is_train=False):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.is_train = is_train
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        
        voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
        self.vector = CamrestVector(voc_file, voc_opp_file)
               
        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        
        self.load(cfg['load'])
        
    def predict(self, state):
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
    
    def load(self, filename):
        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            print('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
