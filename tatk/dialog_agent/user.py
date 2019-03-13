"""User simulator derived from PipelineAgent"""
from .agent import PipelineAgent


class User_Simulator(PipelineAgent):
    """
    Pipeline user simulator
    """
    def __init__(self, nlu_model, policy, nlg_model, mode=0):
        """
        Args:
            nlu_model (NLU): An instance of NLU class.
            # tracker (Tracker): An instance of Tracker class.
            policy (User_Policy): An instance of Policy class.
            nlg_model (NLG): An instance of NLG class.
            mode (int): 0 for utterance level and 1 for dialog act level.
        """
        super().__init__(nlu_model=nlu_model,
                         tracker=None,
                         policy=policy,
                         nlg_model=nlg_model)

        self.mode = mode
        self.current_action = None
        self.policy.init_session()

    def response(self, input, sess=None):
        """
        Generate the response of user.
        Args:
            input: Preorder system output, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
        Returns:
            output (str): User response, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
            session_over (boolean): True to terminate session, else session continues.
            reward (float): The reward given by the user.
        """

        if self.nlu_model is not None:
            sys_act = self.nlu_model.parse(input)
        else:
            sys_act = input
        action, session_over, reward = self.policy.predict(None, sys_act)
        if self.nlg_model is not None:
            output = self.nlg_model.generate(action)
        else:
            output = action

        self.current_action = action

        return output, action, session_over, reward

    def init_session(self):
        """Init the parameters for a new session."""
        self.policy.init_session()
        self.current_action = None

    def init_response(self):
        return {}