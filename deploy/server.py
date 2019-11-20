#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import json
import copy
from deploy.ctrl import ModuleCtrl, SessionCtrl
from deploy.utils import DeployError


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

    def on_register(self, **kwargs):
        ret = {'nlu': 0, 'dst': 0, 'policy': 0, 'nlg': 0}
        try:
            for module_name in ['nlu', 'dst', 'policy', 'nlg']:
                model_id = kwargs.get(module_name, None)
                if isinstance(model_id, str):
                    ret[module_name] = self.modules[module_name].add_used_num(model_id)
        except Exception as e:
            for module_name in ['nlu', 'dst', 'policy', 'nlg']:
                model_id = kwargs.get(module_name, None)
                if isinstance(model_id, str) and ret[module_name] != 0:
                    self.modules[module_name].sub_used_num(model_id)
            raise e

        if ret['nlu'] == 0 and ret['dst'] == 0 and ret['policy'] == 0 and ret['nlg'] == 0:
            raise DeployError('At least one model needs to be started')

        token = self.sessions.new_session(*[kwargs.get(mn, None) for mn in ['nlu', 'dst', 'policy', 'nlg']])

        return {'token': token}

    def on_close(self, token):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)

        sess_info = self.sessions.pop_session(token)
        for module in ['nlu', 'dst', 'policy', 'nlg']:
            self.modules[module].sub_used_num(sess_info[module])
        return {'del_token': token}

    def on_clear_expire(self):
        expire_session = self.sessions.pop_expire_session()
        del_tokens = []
        for (token, sess_info) in expire_session.items():
            del_tokens.append(token)
            for module in ['nlu', 'dst', 'policy', 'nlg']:
                self.modules[module].sub_used_num(sess_info[module])
        return {'del_tokens': del_tokens}

    def on_response(self, token, input_module, data, modified_output={}):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        session_infos = self.sessions.get_session(token)

        ret, session_infos = self.turn(session_infos, input_module, data, modified_output)

        self.sessions.set_session(token, session_infos)
        return ret

    def on_edit_last(self, token, modified_output):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        session_infos = self.sessions.get_session(token)

        if not session_infos['cache']:
            raise DeployError('This is the first turn in this session.')

        last_cache = session_infos['cache'][-1]
        session_infos['cache'] = session_infos['cache'][:-1]
        for (key, value) in last_cache['modified_output'].items():
            modified_output.setdefault(key, value)

        ret, session_infos = self.turn(session_infos, last_cache['input_module'], last_cache['data'], modified_output)

        self.sessions.set_session(token, session_infos)
        return ret

    def turn(self, session_infos, input_module, data, modified_output):
        modules_list = ['nlu', 'dst', 'policy', 'nlg']

        # params
        cur_cache = {name: None for name in modules_list}
        history = []
        if session_infos['cache']:
            cur_cache = {name: session_infos['cache'][-1].get(name, None) for name in modules_list}
            for cache in session_infos['cache']:
                history.append(['user', cache.get('usr', '')])
                history.append(['system', cache.get('sys', '')])

        # process
        new_cache = {name: None for name in modules_list}
        model_ret = {name: None for name in modules_list}
        temp_data = None
        for mod in modules_list:
            if input_module == mod:
                temp_data = data

            if temp_data is not None and session_infos[mod] is not None:
                (model_ret[mod], new_cache[mod]) = self.modules[mod].run(session_infos[mod], cur_cache[mod], not session_infos['cache'],
                                                                         [temp_data, history] if mod == 'nlu' else [temp_data])
                if mod in modified_output.keys():
                    model_ret[mod] = modified_output[mod]

                temp_data = model_ret[mod]
            elif mod == 'policy':
                temp_data = None



        # save cache
        new_cache['usr'] = data if isinstance(data, str) and input_module == 'nlu' else ''
        new_cache['sys'] = model_ret['nlg'] if isinstance(model_ret['nlg'], str) else ''
        new_cache['input_module'] = input_module
        new_cache['data'] = data
        new_cache['modified_output'] = modified_output
        session_infos['cache'].append(copy.deepcopy(new_cache))

        # update history
        history.append(['user', new_cache['usr']])
        history.append(['system', new_cache['sys']])
        model_ret['history'] = history

        return model_ret, session_infos

    def on_rollback(self, token, back_turns=1):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        sess_info = self.sessions.get_session(token)
        sess_info['cache'] = sess_info['cache'][:-back_turns]
        turns = len(sess_info['cache'])
        self.sessions.set_session(token, sess_info)
        return {'current_turns': turns}


if __name__ == '__main__':
    pass
