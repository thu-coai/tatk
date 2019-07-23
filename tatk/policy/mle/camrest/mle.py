# -*- coding: utf-8 -*-
import torch
import os
import json
import zipfile
from tatk.util.file_util import cached_path
from tatk.policy.policy import Policy
from tatk.policy.rlmodule import MultiDiscretePolicy
from tatk.policy.vector.vector_camrest import CamrestVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "mle_policy_camrest.zip")

class MLE(Policy):
    
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/mle_policy_camrest.zip'):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        
        voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
        self.vector = CamrestVector(voc_file, voc_opp_file)
               
        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MLE Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_mle.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
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
