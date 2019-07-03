def default_state():
    state = {'user_action':{}, 'system_action':{}, 'belief_state':{}, 'terminal':False}
    state['belief_state'] = {'address': '',
     'area': '',
     'food': '',
     'name': '',
     'phone': '',
     'pricerange': ''
     }
    return state