"""Natural Language Understanding Interface"""


class NLU:
    """Base class for NLU model."""
    def __init__(self):
        """ Constructor for NLU class. """
        pass

    def parse(self, utterance, sess=None):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        pass
