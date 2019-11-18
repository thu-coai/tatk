#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
      "class_path": "tatk.nlu.svm.camrest.nlu.SVMNLU",  // (not default), Target model class relative path
      "data_set": "camrest",                            // (not default), The data set used by the model
      "ini_params": {"mode": "usr"},                    // (default as {}), The parameters required for the class to be instantiated
      "model_name": "svm",                              // (default as model key), Model name displayed on the front end
      "max_core": 2,                                    // (default as 1), The maximum number of backgrounds allowed for this model to start.
                                                        //                  Recommended to set to 1, or not set.
      "enable": true                                    // (default as true), If false, the system will ignore this configuration

"""

from deploy.utils import MyLock, MySemaphore, DeployError


class ResourceLock(object):
    def __init__(self, value: int = 1):
        self.sema = MySemaphore(value)
        self.lock = MyLock()
        self.used = [0 for _ in range(value)]

    def res_catch(self):
        self.sema.enter()
        with self.lock:
            un_used_idx = self.used.index(0)
            self.used[un_used_idx] = 1
        return un_used_idx

    def res_leave(self, idx):
        with self.lock:
            self.used[idx] = 0
        self.sema.leave()


class ModelCtrl(object):
    def __init__(self, *args, **kwargs):
        # model id
        self.model_id = args[0]

        # running params
        self.model_class = kwargs['class']
        self.ini_params = kwargs.get('ini_params', dict({}))
        self.max_core = kwargs.get('max_core', 1)

        # do not care
        self.class_path = kwargs.get('class_path', '')
        self.model_name = kwargs.get('model_name', '')
        self.data_set = kwargs.get('data_set', '')

        self.opt_lock = MyLock()
        self.used_num = 0

        self.models = [None for _ in range(self.max_core)]
        self.mod_res_lock = ResourceLock(self.max_core)

    def add_used_num(self):
        with self.opt_lock:
            if self.used_num == 0:
                try:
                    self.models = [self.__implement() for _ in range(self.max_core)]
                except Exception as e:
                    raise DeployError('Instantiation failed', model=self.model_id)

            self.used_num += 1

    def sub_used_num(self):
        with self.opt_lock:
            self.used_num -= 1
            self.used_num = 0 if self.used_num < 0 else self.used_num
            if self.used_num == 0:
                for mod in self.models:
                    del mod
                self.models = [None for _ in range(self.max_core)]

    def __implement(self):
        return self.model_class(**self.ini_params)


def test():
    print('hello man!')


if __name__ == '__main__':
    test()
