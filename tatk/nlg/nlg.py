"""Natural Language Generation Interface"""
from abc import ABCMeta, abstractmethod


class NLG(metaclass=ABCMeta):
    """Base class for NLG model."""

    @abstractmethod
    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (dict):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            response (str):
                A natural langauge utterance.
        """
        pass
