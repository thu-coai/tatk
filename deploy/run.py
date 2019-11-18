#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import os
import sys
import json
from flask import Flask, request
from flask import _request_ctx_stack

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
        if fun == 'models':
            ret = ctrl_server.on_models()
        elif fun == 'register':
            ret = ctrl_server.on_register(**params)
        elif fun == 'close':
            ret = ctrl_server.on_close(**params)
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


"""
import time
# ll = Lock()
# ll = GlobalLock()


ll = GlobalSemaphore(2)
# ll = MySemaphore(2)

@app.route('/')
def index():
    print('start')
    with ll:
        print(_request_ctx_stack._local.__ident_func__(), 'bng')

        time.sleep(5)
        '''
        while True:
            pass
        '''

        print(_request_ctx_stack._local.__ident_func__(), 'end')
    print('over')
    return '<h1>hello</h1>'
"""

"""

from concurrent.futures import ThreadPoolExecutor
import threading

pool = ThreadPoolExecutor(max_workers=2)

def work(ids: str):
    print(ids, 'bng')
    time.sleep(5)
    print(ids, 'end')
    return '<h1>hello</h1>'

@app.route('/')
def index():
    f = pool.submit(work, _request_ctx_stack._local.__ident_func__())
    ret = f.result()
    return ret
"""
from gunicorn.app.wsgiapp import run

if __name__ == '__main__':
    # gunicorn deploy.run:app --threads 4
    app.run(host='0.0.0.0', port=dep_conf['net']['port'])
