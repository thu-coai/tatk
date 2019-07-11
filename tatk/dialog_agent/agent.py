"""Dialog agent interface and classes."""
from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
    """Interface for dialog agent classes."""

    @abstractmethod
    def response(self, observation):
        """Generate agent response given user input.

        The data type of input and response can be either str or dict, condition on the form of agent.

        Example:
            If the agent is a pipeline agent with NLU, DST and Policy, then type(input) == str and
            type(response) == dict.
        Args:
            observation (str or dict):
                The input to the agent.
        Returns:
            response (str or dict):
                The response generated by the agent.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Reset the class variables to prepare for a new session."""
        pass


class PipelineAgent(Agent):
    """Pipeline dialog agent base class, including NLU, DST, Policy and NLG.

    The combination modes of pipeline agent modules are flexible. The only thing you have to make sure is that
    the API of agents are matched.

    Example:
        If agent A is (nlu, tracker, policy), then the agent B should be like (tracker, policy, nlg) to ensure API
        matching.
    The valid module combinations are as follows:
           =====   =====    ======  ===     ==      ===
            NLU     DST     Policy  NLG     In      Out
           =====   =====    ======  ===     ==      ===
            \+      \+        \+    \+      nl      nl
             o      \+        \+    \+      da      nl
             o      \+        \+     o      da      da
            \+      \+        \+     o      nl      da
             o       o        \+     o      da      da
           =====   =====    ======  ===     ==      ===
    """

    def __init__(self, nlu_model, tracker, policy, nlg_model):
        """The constructor of PipelineAgent class.

        Here are some special combination cases:

            1. If you use word-level DST (such as Neural Belief Tracker), you should set the nlu_model paramater \
             to None. The agent will combine the modules automitically.

            2. If you want to aggregate DST and Policy as a single module, set tracker to None.

        Args:
            nlu_model (NLU):
                The natural langauge understanding module of agent.

            tracker (Tracker):
                The dialog state tracker of agent.

            policy (Policy):
                The dialog policy module of agent.

            nlg_model (NLG):
                The natural langauge generator module of agent.
        """
        super(PipelineAgent, self).__init__()
        self.nlu_model = nlu_model
        self.tracker = tracker
        self.policy = policy
        self.nlg_model = nlg_model
        self.init_session()

    def response(self, observation):
        """Generate agent response using the agent modules."""
        # get dialog act
        if self.nlu_model is not None:
            dialog_act = self.nlu_model.predict(observation)
        else:
            dialog_act = observation

        # get action
        if self.tracker is not None:
            state = self.tracker.update(dialog_act)
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(dialog_act)

        # get model response
        if self.nlg_model is not None:
            model_response = self.nlg_model.generate(action)
        else:
            model_response = action

        return model_response

    def is_terminal(self):
        if hasattr(self.policy, 'is_terminal'):
            return self.policy.is_terminal()
        return None

    def get_reward(self):
        if hasattr(self.policy, 'get_reward'):
            return self.policy.get_reward()
        return None

    def init_session(self):
        """Init the attributes of DST and Policy module."""
        if self.tracker is not None:
            self.tracker.init_session()
        if self.policy is not None:
            self.policy.init_session()
