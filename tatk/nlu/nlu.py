"""Natural language understanding interface."""
from abc import ABCMeta, abstractmethod


class NLU(metaclass=ABCMeta):
    """NLU module interface."""

    @abstractmethod
    def predict(self, utterance, context=list()):
        """Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (string):
                A natural language utterance.
            context (list of string):
                Previous utterances.

        Returns:
            output (list of tuples):
                The dialog act of utterance.
        """
        return self.predict_batch([utterance], [context])

    @abstractmethod
    def predict_batch(self, batch_utterance, batch_context=list()):
        """Predict the dialog act of a batch of natural language utterances.

        Args:
            batch_utterance (list of string):
                Natural language utterances.
            batch_context (list of list of string):
                Previous utterances.

        Returns:
            output (list of list of tuples):
                The dialog act of utterances.
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
