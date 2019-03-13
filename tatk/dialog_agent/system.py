"""Dialog System derived from PipelineAgent"""
from .agent import PipelineAgent


class Dialog_System(PipelineAgent):
    """
    Pipeline dialog system
    """
    def __init__(self, nlu_model, tracker, policy, nlg_model, mode=0):
        """
        Args:
            nlu_model (NLU): An instance of NLU class.
            tracker (Tracker): An instance of Tracker class.
            policy (Sys_Policy): An instance of Policy class.
            nlg_model (NLG): An instance of NLG class.
            mode (int): 0 for utterance level (without nlu);
                        1 for dialog act level (without nlu);
                        2 for utterance level (with nlu).
        """
        super().__init__(nlu_model=nlu_model,
                         tracker=tracker,
                         policy=policy,
                         nlg_model=nlg_model)
        self.mode = mode

    def response(self, input, sess):
        """
        Generate the response of system bot.
        Args:
            input (dict or str): Preorder user output. The variable type depends on the user model's configuration,
                    i.e., if the user.nlg is not None, type(input) == str; else if user.nlg is None, type(input) = dict.
            sess (Session):
        Returns:
            output (str or dict): The output of system agent.
            action (dict): The dialog act of the output of system agent.
        """

        if self.nlu_model is not None:
            user_act = self.nlu_model.predict(input)
        else:
            user_act = input
        history = self.tracker.state['history']
        if len(history) == 0:  # init state
            state = self.tracker.state
        else:
            state = self.tracker.update(sess, user_act)  # state is exactly self.tracker.state
        action = self.policy.predict(state)
        if self.nlg_model is not None:
            output = self.nlg_model.generate(action)
        else:
            output = action

        return output, action

    def init_session(self):
        """Init the parameters for a new session."""
        self.tracker.init_session()
        self.policy.init_session()