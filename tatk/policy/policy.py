"""Policy Interface"""


class Policy():
    """Base class for policy model."""

    def __init__(self, is_train=False):
        """ Constructor for Policy class. """
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

