# Rule policy
Rule policy is a rule based dialog policy for Multiwoz dataset. It takes a dialog state as input and generates system's dialog act.

# How to use
Example:

```python
from tatk.policy.rule.camrest.rule_based_camrest_bot import RuleBasedCamrestBot
sys_policy = RuleBasedCamrestBot()

# Policy takes dialog state as input. Please refer to tatk.util.camrest.state

state = {'user_action': {'inform': [['name', 'Chiquito Restaurant Bar'],
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
 'terminal': False,
 'history': []}

# Please call `init_session` before a new session, this clears policy's history info.
sys_policy.init_session()
    
# method `predict` takes state output from tracker, and generates system's dialog act.
sys_da = sys_policy.predict(state)}
```