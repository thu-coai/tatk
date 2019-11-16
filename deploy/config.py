#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load deployment settings from config file
"""
import os
import sys
import json


def load_config_file(filepath: str = None) -> dict:
    """
    load config setting from json file
    :param filepath: str, dest config file path
    :return: dict,
    """
    if not isinstance(filepath, str):
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dep_config.json'))

    # load
    with open(filepath, 'r', encoding='UTF-8') as f:
        conf = json.load(f)
    assert isinstance(conf, dict), 'Incorrect format in config file \'%s\'' % filepath

    # check sections
    for sec in ['net', 'nlu', 'dst', 'policy', 'nlg']:
        assert sec in conf.keys(), 'Missing \'%s\' section in config file \'%s\'' % (sec, filepath)

    # check net
    assert isinstance(conf['net'].get('port', None), int), 'Incorrect key \'net\'->\'port\' in config file \'%s\'' % filepath
    conf['net']['app_name'] = conf['net'].get('app_name', '')
    assert isinstance(conf['net'].get('app_name', None), str), 'Incorrect key \'net\'->\'app_name\' in config file \'%s\'' % filepath

    # check model sections
    for sec in ['nlu', 'dst', 'policy', 'nlg']:
        conf[sec] = {key: info for (key, info) in conf[sec].items() if info.get('enable', False)}
        assert conf[sec], '\'%s\' section can not be empty in config file \'%s\'' % (sec, filepath)

    return conf


def map_class(cls_path: str):
    """
    Map to class via package text path
    :param cls_path: str, path with `tatk` project directory as relative path, separator with `,`
                            E.g  `tatk.nlu.svm.camrest.nlu.SVMNLU`
    :return: class
    """
    pkgs = cls_path.split('.')
    root_pkg = '.'.join(pkgs[:-1])
    pkgs = pkgs[1:]
    cls = __import__(root_pkg)
    for pkg in pkgs:
        cls = getattr(cls, pkg)
    return cls


def get_config(filepath: str = None) -> dict:
    """
    The configuration file is used to create all the information needed for the deployment,
    and the necessary security monitoring has been performed, including the mapping of the class.
    :param filepath: str, dest config file path
    :return: dict
    """
    # load settings
    conf = load_config_file(filepath)

    # add project root dir
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    # reflect class
    # NLU
    from tatk.nlu import NLU
    for (name, infos) in conf['nlu'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, NLU), '\'%s\' is not a %s class' % (cls_path, 'nlu')
        conf['nlu'][name]['class'] = cls

    # DST
    from tatk.dst import Tracker
    for (name, infos) in conf['dst'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, Tracker), '\'%s\' is not a %s class' % (cls_path, 'dst')
        conf['dst'][name]['class'] = cls

    # Policy
    from tatk.policy import Policy
    for (name, infos) in conf['policy'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, Policy), '\'%s\' is not a %s class' % (cls_path, 'policy')
        conf['policy'][name]['class'] = cls

    # NLG
    from tatk.nlg import NLG
    for (name, infos) in conf['nlg'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, NLG), '\'%s\' is not a %s class' % (cls_path, 'nlg')
        conf['nlg'][name]['class'] = cls

    return conf


if __name__ == '__main__':
    # test
    cfg = get_config()
    nlu_mod = cfg['nlu']['svm-cam']['class'](**cfg['nlu']['svm-cam']['ini_params'])
    dst_mod = cfg['dst']['rule-cam']['class'](**cfg['dst']['rule-cam']['ini_params'])
    plc_mod = cfg['policy']['mle-cam']['class'](**cfg['policy']['mle-cam']['ini_params'])
    nlg_mod = cfg['nlg']['tmp-cam-usr-manual']['class'](**cfg['nlg']['tmp-cam-usr-manual']['ini_params'])
