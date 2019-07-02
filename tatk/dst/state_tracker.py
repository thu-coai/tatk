"""Dialog State Tracker Interface"""
from abc import ABCMeta, abstractmethod

class Tracker(metaclass=ABCMeta):
    """Base class for dialog state tracker models."""

    @abstractmethod
    def update(self, dialog_act):
        """ Update the internal dialog state variable.

        Args:
            dialog_act (str or dict): The type is str when Tracker is word-level (such as NBT), and dict when it is
                    DA-level.
        Returns:
            new_state (tuple): Updated dialog state, with the same form of previous state.
        """
        pass

    @abstractmethod
    def init_session(self):
        pass
