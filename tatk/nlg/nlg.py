"""Natural Language Generation Interface"""
from abc import abstractmethod
from tatk.util.module import Module


class NLG(Module):
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
    def get_name(self):
        return super(NLG, self).get_name() + '-' + 'NLG'
