"""Natural language understanding interface."""
from abc import abstractmethod
from tatk.util.module import Module


class NLU(Module):
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

    # @abstractmethod
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
