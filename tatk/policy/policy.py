"""Policy Interface"""
from abc import ABC, abstractmethod


class Policy(ABC):
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

    @abstractmethod
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

    @abstractmethod
    def train(self, *args, **kwargs):
        """Model training entry point"""
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        """Model testing entry point"""
        pass

    def from_cache(self, *args, **kwargs):
        """restore internal state for multi-turn dialog"""
        return None

    def to_cache(self, *args, **kwargs):
        """save internal state for multi-turn dialog"""
        return None

    def init_session(self):
        """Init the class variables for a new session."""
        pass
