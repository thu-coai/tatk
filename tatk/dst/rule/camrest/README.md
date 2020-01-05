# RuleDST
RuleDST is a rule based dialog state tracker which trivially updates new values from NLU result to states.

# How to use
Example:

```python
from tatk.dst.rule.camrest.state_tracker import RuleDST


tracker = RuleDST()

# Action is a dict. Its keys are strings(type) and its values are list of lists. 
# The type may be one of ('inform', 'request', 'nooffer').

# For example, the action below has a key "inform".
# Each list in the value of "inform" is a slot-value pair. "name" is slot and "Chiquito Restaurant Bar" is value. "pricerange" is slot and "expensive" is value.
# Note! Keys and slots must be lowercase.
action = {'inform': [['name', 'Chiquito Restaurant Bar'],
    ['pricerange', 'expensive'],
    ['area', 'south'],
    ['food', 'mexican']]}

# method `update` updates the attribute `state` of tracker, and returns it.
state = tracker.update(action)
assert state == tracker.state
assert state == {'user_action': {'inform': [['name', 'Chiquito Restaurant Bar'],
   ['pricerange', 'expensive'],
   ['area', 'south'],
   ['food', 'mexican']]},
 'system_action': {},
 'belief_state': {'address': '',
  'area': 'south',
  'food': 'mexican',
  'name': 'Chiquito Restaurant Bar',
  'phone': '',
  'pricerange': 'expensive'},
 'request_state': {},
 'terminated': False,
 'history': []}

# Please call `init_session` before a new dialog. This initializes the attribute `state` of tracker with a default state, which `tatk.util.camrest.state.default_state` returns. But You needn't call it before the first dialog, because tracker gets a default state in its constructor.
tracker.init_session()
action = {} # empty dict is OK
state = tracker.update(action)
assert state == {'user_action': {},
 'system_action': {},
 'belief_state': {'address': '',
  'area': 'south',
  'food': 'mexican',
  'name': 'Chiquito Restaurant Bar',
  'phone': '',
  'pricerange': 'expensive'},
 'request_state': {},
 'terminated': False,
 'history': []}

```
