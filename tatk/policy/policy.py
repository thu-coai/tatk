"""Policy Interface"""


class Policy:
    def __init__(self):
        pass

    def predict(self, state, sess=None):
        pass

    def init_session(self):
        pass


class User_Policy(Policy):
    """Base model for user policy model."""
    def __init__(self):
        """ Constructor for User_Policy class. """
        super().__init__()
        pass

    def predict(self, state, sess=None):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
            sess (tuple): Session
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by the user.
        """
        pass
    
    def init_session(self):
        """
        Restore after one session
        """
        pass


class Sys_Policy(Policy):
    """Base class for system policy model."""

    def __init__(self, is_train=False):
        """ Constructor for Sys_Policy class. """
        super().__init__()
        self.is_train = is_train
        pass

    def predict(self, state, sess=None):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        pass

    def init_session(self):
        """
        Restore after one session
        """
        pass
