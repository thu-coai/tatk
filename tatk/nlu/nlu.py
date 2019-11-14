"""Natural language understanding interface."""
from abc import ABC, abstractmethod


class NLU(ABC):
    """NLU module interface."""

    def predict(self, utterance, context=list()):
        """Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (string):
                A natural language utterance.
            context (list of string):
                Previous utterances.

        Returns:
            action (list of tuples):
                The dialog act of utterance.
        """
        return self.predict_batch([utterance], [context])[0]

    @abstractmethod
    def predict_batch(self, batch_utterance, batch_context=list()):
        """Predict the dialog acts of a batch of natural language utterances.

        Args:
            batch_utterance (list of string):
                Natural language utterances.
            batch_context (list of list of string):
                Previous utterances.

        Returns:
            batch_action (list of list of tuples):
                The dialog acts of utterances.
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
