"""Policy Interface"""
from abc import abstractmethod
from tatk.util.module import Module


class Policy(Module):
    """Base class for policy model."""

    def predict(self, state):
        """Predict the next agent action given dialog state.
        
        Args:
            state (tuple or dict):
                when the DST and Policy module are separated, the type of state is tuple.
                else when they are aggregated together, the type of state is dict (dialog act).
        Returns:
            action (list of tuples):
                The next dialog action.
        """
        return self.predict_batch([state])[0]

    # @abstractmethod
    def predict_batch(self, batch_state):
        """Predict actions given a batch of dialog states.

        Args:
            batch_state (list of tuple or dict):
                when the DST and Policy module are separated, the type of state is tuple.
                else when they are aggregated together, the type of state is dict (dialog act).
        Returns:
            batch_action (list of list of tuples):
                The next dialog action.
        """
        pass
