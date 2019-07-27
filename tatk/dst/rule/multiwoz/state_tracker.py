import copy
import json
import os

from tatk.util.multiwoz.state import default_state
from tatk.dst.rule.multiwoz.dst_util import normalize_value
from tatk.dst.state_tracker import Tracker
from tatk.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA


class RuleDST(Tracker):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``tatk.util.camrest.state.default_state`` returns a default state.
        value_dict(dict):
            It helps check whether ``user_act`` has correct content.
    """
    def __init__(self):
        Tracker.__init__(self)
        self.state = default_state()
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path,'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

    def update(self, user_act=None):
        # print('------------------{}'.format(user_act))
        if type(user_act) is not dict:
            raise Exception('Expect user_act to be <class \'dict\'> type but get {}.'.format(type(user_act)))
        previous_state = self.state
        new_belief_state = copy.deepcopy(previous_state['belief_state'])
        new_request_state = copy.deepcopy(previous_state['request_state'])
        for domain_type in user_act.keys():
            domain, tpe = domain_type.lower().split('-')
            if domain in ['unk', 'general', 'booking']:
                continue
            if tpe == 'inform':
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if k is None:
                        continue
                    try:
                        assert domain in new_belief_state
                    except:
                        raise Exception('Error: domain <{}> not in new belief state'.format(domain))
                    domain_dic = new_belief_state[domain]
                    assert 'semi' in domain_dic
                    assert 'book' in domain_dic

                    if k in domain_dic['semi']:
                        nvalue = normalize_value(self.value_dict, domain, k, v)
                        # if nvalue != v:
                        #     _log('domain {} slot {} value {} -> {}'.format(domain, k, v, nvalue))
                        new_belief_state[domain]['semi'][k] = nvalue
                    elif k in domain_dic['book']:
                        new_belief_state[domain]['book'][k] = v
                    elif k.lower() in domain_dic['book']:
                        new_belief_state[domain]['book'][k.lower()] = v
                    elif k == 'trainID' and domain == 'train':
                        new_belief_state[domain]['book'][k] = normalize_value(self.value_dict, domain, k, v)
                    else:
                        # raise Exception('unknown slot name <{}> of domain <{}>'.format(k, domain))
                        with open('unknown_slot.log', 'a+') as f:
                            f.write('unknown slot name <{}> of domain <{}>\n'.format(k, domain))
            elif tpe == 'request':
                for k, v in user_act[domain_type]:
                    k = REF_SYS_DA[domain.capitalize()].get(k, k)
                    if domain not in new_request_state:
                        new_request_state[domain] = {}
                    if k not in new_request_state[domain]:
                        new_request_state[domain][k] = 0

        new_state = copy.deepcopy(previous_state)
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        new_state['user_action'] = user_act

        self.state = new_state
        
        return self.state

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.multiwoz.state.default_state`` returns."""
        self.state = default_state()
