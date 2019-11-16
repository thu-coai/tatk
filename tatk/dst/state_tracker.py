"""Dialog State Tracker Interface"""
from abc import abstractmethod
from tatk.util.module import Module


class Tracker(Module):
    """Base class for dialog state tracker models."""

    @abstractmethod
    def update(self, action):
        """ Update the internal dialog state variable.

        Args:
            action (str or list of tuples):
                The type is str when Tracker is word-level (such as NBT), and list of tuples when it is DA-level.
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        return self.update_batch([action])[0]

    @abstractmethod
    def update_batch(self, batch_action):
        """ Update the internal dialog state variable.

        Args:
            batch_action (list of str or list of list of tuples):
                The type is list of str when Tracker is word-level (such as NBT), and list of list of tuples when it is DA-level.
        Returns:
            batch_new_state (list of dict):
                Updated dialog states, with the same form of previous states.
        """
        pass

    @abstractmethod
    def get_name(self):
        return super(Tracker, self).get_name() + '-' + 'DST'
