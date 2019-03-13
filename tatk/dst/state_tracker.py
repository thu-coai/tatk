"""Dialog State Tracker Interface"""


class Tracker:
    """Base class for dialog state tracker models."""
    def __init__(self):
        pass

    def update(self, user_act=None, sess=None):
        """
        Update dialog state.
        Args:
            sess (Session Object):
        Returns:
            new_state (tuple): Updated dialog state, with the same form of previous state.
        """
        pass

    def init_session(self):
        pass
