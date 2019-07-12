from tatk.dst.state_tracker import Tracker
from tatk.util.camrest.state import default_state
import copy


class RuleDST(Tracker):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``tatk.util.camrest.state.default_state`` returns a default state.
    """
    def __init__(self):
        super().__init__()
        self.state = default_state()

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.camrest.state.default_state`` returns."""
        self.state = default_state()

    def update(self, user_act=None):
        if not isinstance(user_act, dict):
            raise TypeError('Expect user_act to be <class \'dict\'> type but get {}.'.format(type(user_act)))
        self.state['user_action'] = user_act
        for key, list_ in user_act.items():
            if key == "nooffer":
                continue
            elif key == "inform":
                for slot, value in list_:
                    if slot not in self.state['belief_state']:
                        continue
                    self.state['belief_state'][slot] = value
            elif key == "request":
                for slot, _ in list_:
                    self.state['request_state'][slot] = 0
        return copy.deepcopy(self.state)
