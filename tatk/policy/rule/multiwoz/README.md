# Rule policy
Rule policy is a rule based dialog policy for Multiwoz dataset. It takes a dialog state as input and generates system's dialog act.

# How to use
Example:

```python
from tatk.policy.rule.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot
sys_policy = RuleBasedMultiwozBot()

# Policy takes dialog state as input. Please refer to tatk.util.multiwoz.state

state = {'user_action': {'Hotel-Inform': [['Area', 'east'], ['Stars', '4']]},
'system_action': {},
 'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                  'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                            'semi': {'name': '',
                                     'area': 'east',
                                     'parking': '',
                                     'pricerange': '',
                                     'stars': '4',
                                     'internet': '',
                                     'type': ''}},
                  'attraction': {'book': {'booked': []},
                                 'semi': {'type': '', 'name': '', 'area': ''}},
                  'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                  'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                  'taxi': {'book': {'booked': []},
                           'semi': {'leaveAt': '',
                                    'destination': '',
                                    'departure': '',
                                    'arriveBy': ''}},
                  'train': {'book': {'booked': [], 'people': ''},
                            'semi': {'leaveAt': '',
                                     'destination': '',
                                     'day': '',
                                     'arriveBy': '',
                                     'departure': ''}}},
 'request_state': {},
 'terminal': False,
 'history': []}

# Please call `init_session` before a new session, this clears policy's history info.
sys_policy.init_session()
    
# method `predict` takes state output from tracker, and generates system's dialog act.
sys_da = sys_policy.predict(state)
```