#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import json
import copy
from deploy.ctrl.module import ModuleCtrl
from deploy.ctrl.session import SessionCtrl


class ServerCtrl(object):
    def __init__(self, **kwargs):
        self.net_conf = copy.deepcopy(kwargs['net'])
        self.module_conf = {
            'nlu': copy.deepcopy(kwargs['nlu']),
            'dst': copy.deepcopy(kwargs['dst']),
            'policy': copy.deepcopy(kwargs['policy']),
            'nlg': copy.deepcopy(kwargs['nlg'])
        }
        self.modules = {mdl: ModuleCtrl(mdl, self.module_conf[mdl]) for mdl in self.module_conf.keys()}
        self.sessions = SessionCtrl(expire_sec=self.net_conf['session_time_out'])

    def on_models(self):
        ret = {}
        for module_name in ['nlu', 'dst', 'policy', 'nlg']:
            ret[module_name] = {}
            for model_id in self.module_conf[module_name].keys():
                ret[module_name][model_id] = {key: self.module_conf[module_name][model_id][key] for key in
                                              ['class_path', 'data_set', 'ini_params', 'model_name']}
                ret[module_name][model_id]['ini_params'] = json.dumps(ret[module_name][model_id]['ini_params'])
        return ret

    def on_register(self, nlu, dst, policy, nlg):
        self.modules['nlu'].add_used_num(nlu)
        self.modules['dst'].add_used_num(dst)
        self.modules['policy'].add_used_num(policy)
        self.modules['nlg'].add_used_num(nlg)

        self.modules['nlu'].sub_used_num(nlu)
        self.modules['dst'].sub_used_num(dst)
        self.modules['policy'].sub_used_num(policy)
        self.modules['nlg'].sub_used_num(nlg)
        return {}

    def on_close(self, token):
        return {}


def test():
    print('hello man!')


if __name__ == '__main__':
    test()
