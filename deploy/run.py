#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import os
import sys
import json
from flask import Flask, request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from deploy.config import get_config
from deploy.utils import DeployError
from deploy.server import ServerCtrl

# load config file
dep_conf = get_config()
app_name = dep_conf['net']['app_name'].strip('/')
app_name = '/' + app_name if app_name else ''

# server control obj
ctrl_server = ServerCtrl(**dep_conf)

# flask app
app = Flask(__name__)


def get_params_from_request(gp_reqt):
    in_dict = {}
    if gp_reqt.method == 'POST':
        if 'application/json' in gp_reqt.headers.environ['CONTENT_TYPE']:
            in_dict = gp_reqt.json
        elif 'application/x-www-form-urlencoded' in gp_reqt.headers.environ['CONTENT_TYPE']:
            in_dict = gp_reqt.form.to_dict()
    elif gp_reqt.method == 'GET':
        in_dict = gp_reqt.args.to_dict()
    return in_dict


@app.route('%s/<fun>' % app_name, methods=['GET', 'POST'])
def net_function(fun):
    params = get_params_from_request(request)
    ret = {}
    try:
        # clear expire session every time
        ctrl_server.on_clear_expire()

        if fun == 'models':
            ret = ctrl_server.on_models()
        elif fun == 'register':
            ret = ctrl_server.on_register(**params)
        elif fun == 'close':
            ret = ctrl_server.on_close(**params)
        elif fun == 'clear_expire':
            ret = ctrl_server.on_clear_expire()
        elif fun == 'response':
            ret = ctrl_server.on_response(**params)
        elif fun == 'rollback':
            ret = ctrl_server.on_rollback(**params)
        else:
            raise DeployError('Unknow funtion \'%s\'' % fun)
    except Exception as e:
        err_msg = 'There are some errors in the operation.'
        if isinstance(e, DeployError):
            err_msg = str(e)
        elif isinstance(e, TypeError):
            err_msg = 'Input parameters incorrect for function \'%s\'.' % fun
        ret = {'status': 'error', 'error_msg': err_msg}
    else:
        ret.setdefault('status', 'ok')
    finally:
        ret = json.dumps(ret, ensure_ascii=False)
    return ret


if __name__ == '__main__':
    # gunicorn deploy.run:app --threads 4
    app.run(host='0.0.0.0', port=dep_conf['net']['port'])
