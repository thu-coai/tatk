"""Base class for agent"""


class BaseAgent:
    """
    Base dialog agent class.
    """

    def __init__(self):
        pass


class PipelineAgent(BaseAgent):
    """
    Pipeline dialog agent base class, includes NLU, DST, Policy, NLG.
    """
    def __init__(self, nlu_model, tracker, policy, nlg_model):
        super().__init__()
        self.nlu_model = nlu_model
        self.tracker = tracker
        self.policy = policy
        self.nlg_model = nlg_model

    def response(self, input, sess):
        pass


class End2EndAgent(BaseAgent):
    """
    End2End dialog agent base class.
    """
    def __init__(self):
        super().__init__()

    def response(self, input, sess):
        pass
