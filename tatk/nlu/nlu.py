"""Natural language understanding interface."""
from abc import ABCMeta, abstractmethod


class NLU(metaclass=ABCMeta):
    """NLU module interface."""

    @abstractmethod
    def predict(self, utterance):
        """Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (str):
                A natural language utterance.
        Returns:
            output (dict):
                The dialog act of utterance.
        """
        pass
