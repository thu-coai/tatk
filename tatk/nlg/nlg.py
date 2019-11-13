"""Natural Language Generation Interface"""
from abc import ABC, abstractmethod


class NLG(ABC):
    """Base class for NLG model."""

    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (list of tuples):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            utterance (str):
                A natural langauge utterance.
        """
        return self.generate_batch([action])[0]

    @abstractmethod
    def generate_batch(self, batch_action):
        """Generate natural language utterances conditioned on a batch of dialog acts.

        Args:
            batch_action (list of list of tuples):
                A batch of dialog acts

        Returns:
            batch_utterance (list of string):
                Natural language utterances.
        """
        return [[]]

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
