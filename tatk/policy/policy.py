"""Policy Interface"""
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):
    """Base class for policy model."""
	
    @abstractmethod
    def predict(self, state):
        """Predict the next agent action given dialog state.
        
        Args:
            state (tuple or dict): when the DST and Policy module are separated, the type of state is tuple.
                    else when they are aggregated together, the type of state is dict (dialog act).
        Returns:
            action (dict): The next dialog action.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the class variables for a new session."""
        pass

