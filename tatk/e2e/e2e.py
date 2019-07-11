"""E2E Interface"""
from abc import ABCMeta, abstractmethod


class E2E(metaclass=ABCMeta):
    """Base class for End2End model."""

    @abstractmethod
    def predict(self, utterance):
        """Predict the next agent action given dialog state.

        Args:
            utterance (str): A natural language utterance input.
        Returns:
            response (str): A natural language utterance output.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the class variables for a new session."""
        pass

