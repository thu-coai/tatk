"""Natural Language Generation Interface"""


class NLG:
    """Base class for NLG model."""
    def __init__(self):
        """ Constructor for NLG class. """
        pass

    def generate(self, dialog_act, sess=None):
        """
        Generate a natural language utterance conditioned on the dialog act.
        Args:
            dialog_act (tuple): Dialog act, with the form of (act_type, {slot_name_1: value_1,
                    slot_name_2, value_2, ...})
        Returns:
            response (str): A natural langauge utterance.
        """
        pass