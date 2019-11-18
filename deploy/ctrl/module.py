#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control the invocation of all modules

{
  "class_path": "tatk.nlu.svm.camrest.nlu.SVMNLU",
  "data_set": "camrest",
  "ini_params": {"mode": "usr"},
  "model_name": "svm",
  "max_core": 2,
  "class": cls,

}
"""
import copy
from deploy.ctrl.model import ModelCtrl
from deploy.utils import DeployError


class ModuleCtrl(object):
    def __init__(self, module_name: str, infos: dict):
        self.module_name = module_name
        self.infos = copy.deepcopy(infos)
        self.models = {mid: ModelCtrl(mid, **self.infos[mid]) for mid in self.infos.keys()}

    def add_used_num(self, model_id: str):
        try:
            self.models[model_id].add_used_num()
        except TypeError:
            raise DeployError('Unknow model id \'%s\'' % model_id, module=self.module_name)

    def sub_used_num(self, model_id: str):
        try:
            self.models[model_id].sub_used_num()
        except TypeError:
            raise DeployError('Unknow model id \'%s\'' % model_id, module=self.module_name)


if __name__ == '__main__':
    from deploy.config import get_config

    conf = get_config()
    aaa = ModuleCtrl(**conf)
