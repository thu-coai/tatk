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

    def on_response(self, token, input_module, data):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        sess_info = self.sessions.get_session(token)
        isfirst = not sess_info['cache']
        history = []
        if isfirst:
            cache_nlu, cache_dst, cache_plc, cache_nlg = None, None, None, None
        else:
            last_cache = sess_info['cache'][-1]
            cache_nlu = last_cache.get('nlu', None)
            cache_dst = last_cache.get('dst', None)
            cache_plc = last_cache.get('policy', None)
            cache_nlg = last_cache.get('nlg', None)
            for cache in sess_info['cache']:
                history.append(['user', cache.get('usr', '')])
                history.append(['system', cache.get('sys', '')])

        ret_nlu, ret_dst, ret_plc, ret_nlg = None, None, None, None
        new_cache_nlu, new_cache_dst, new_cache_plc, new_cache_nlg = None, None, None, None

        # NLU
        if input_module == 'nlu':
            in_nlu = data
            if sess_info['nlu'] is not None:
                (ret_nlu, new_cache_nlu) = self.modules['nlu'].run(sess_info['nlu'], cache_nlu, isfirst, [in_nlu, history])
                out_nlu = ret_nlu
            else:
                out_nlu = in_nlu

        # DST
        if input_module in ['nlu', 'dst']:
            in_dst = out_nlu if input_module == 'nlu' else data
            if sess_info['dst'] is not None:
                (ret_dst, new_cache_dst) = self.modules['dst'].run(sess_info['dst'], cache_dst, isfirst, [in_dst])
                out_dst = ret_dst
            else:
                out_dst = in_dst

        # POLICY
        if input_module in ['nlu', 'dst', 'policy']:
            in_plc = out_dst if input_module in ['nlu', 'dst'] else data
            if sess_info['policy'] is not None:
                (ret_plc, new_cache_plc) = self.modules['policy'].run(sess_info['policy'], cache_plc, isfirst, [in_plc])
                out_plc = ret_plc
            else:
                out_plc = None

        # NLG
        in_nlg = out_plc if input_module in ['nlu', 'dst', 'policy'] else data
        if sess_info['nlg'] is not None and in_nlg is not None:
            (ret_nlg, new_cache_nlg) = self.modules['nlg'].run(sess_info['nlg'], cache_nlg, isfirst, [in_nlg])

        # save cache
        new_cache = {
            'nlu': new_cache_nlu, 'dst': new_cache_dst, 'policy': new_cache_plc, 'nlg': new_cache_nlg,
            'usr': data if isinstance(data, str) and input_module == 'nlu' else '',
            'sys': ret_nlg if isinstance(ret_nlg, str) else ''
        }
        sess_info['cache'].append(copy.deepcopy(new_cache))
        self.sessions.set_session(token, sess_info)

        history.append(['user', new_cache.get('usr', '')])
        history.append(['system', new_cache.get('sys', '')])

        return {'nlu': ret_nlu, 'dst': ret_dst, 'policy': ret_plc, 'nlg': ret_nlg, 'history': history}

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
