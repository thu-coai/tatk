def default_state():
    state = {'user_action': {}, 'system_action': {}, 'belief_state': {}, 'request_state': {}, 'terminal': False, 'history': []}
    state['belief_state'] = {'address': '',
                             'area': '',
                             'food': '',
                             'name': '',
                             'phone': '',
                             'pricerange': ''
                             }
    return state
