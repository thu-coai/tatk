"""Dialog controller classes."""
from abc import ABCMeta, abstractmethod
import random
import tatk.e2e.rnn_rollout.utils as utils


class Session(metaclass=ABCMeta):
    """Base dialog session controller, which manages the agents to conduct a complete dialog session.
    """

    @abstractmethod
    def next_agent(self):
        """Decide the next agent to generate a response.

        In this base class, this function returns the index randomly.

        Returns:
            next_agent (Agent): The index of the next agent.
        """
        pass

    @abstractmethod
    def next_response(self, observation):
        """Generated the next response.
        
        Args:
            observation (str or dict): The agent observation of next agent.
        Returns:
            response (str or dict): The agent's response.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the agent variables for a new session."""
        pass


class BiSession(Session):
    """The dialog controller which aggregates several agents to conduct a complete dialog session.

    Attributes:
        sys_agent (Agent):
            system dialog agent.

        user_agent (Agent):
            user dialog agent.

        kb_query (KBquery):
            knowledge base query tool.

        dialog_history (list):
            The dialog history, formatted as [[user_uttr1, sys_uttr1], [user_uttr2, sys_uttr2], ...]
    """

    def __init__(self, sys_agent, user_agent, kb_query):
        """
        Args:
            sys_agent (Agent):
                An instance of system agent.

            user_agent (Agent):
                An instance of user agent.

            kb_query (KBquery):
                An instance of database query tool.
        """
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.kb_query = kb_query

        self.dialog_history = []
        self.__turn_indicator = 0

        self.init_session()

    def next_agent(self):
        """The user and system agent response in turn."""
        if self.__turn_indicator % 2 == 0:
            next_agent = self.user_agent
        else:
            next_agent = self.sys_agent
        self.__turn_indicator += 1
        return next_agent

    def next_response(self, observation):
        next_agent = self.next_agent()
        response = next_agent.response(observation)
        return response

    def next_turn(self, last_observation):
        """Conduct a new turn of dialog, which consists of the system response and user response.

        The variable type of responses can be either 1) str or 2) dialog act, depends on the dialog mode settings of the
        two agents which are supposed to be the same.
        
        Args:
            last_observation:
                Last agent response.
        Returns:
            sys_response:
                The response of system.

            user_response:
                The response of user simulator.

            session_over (boolean):
                True if session ends, else session continues.

            reward (float):
                The reward given by the user.
        """
        user_response = self.next_response(last_observation)
        session_over = self.user_agent.is_terminal()
        reward = self.user_agent.get_reward()
        sys_response = self.next_response(user_response)

        return sys_response, user_response, session_over, reward

    def train_policy(self):
        """
        Train the parameters of system agent policy.
        """
        self.sys_agent.policy.train()

    def init_session(self):
        self.sys_agent.init_session()
        self.user_agent.init_session()

class DealornotSession(Session):
    """A special session for Deal or Not dataset, which is a object dividing negotiation task."""
    def __init__(self, alice, bob):
        self.alice = alice
        self.bob = bob
        self.__turn_indicator = 0
        self.init_session()
        self.current_agent = None
        self.dialog_history = []

    def next_agent(self):
        """Alice and Bob agents response in turn."""
        if self.__turn_indicator % 2 == 0:
            next_agent = self.alice
        else:
            next_agent = self.bob
        self.__turn_indicator += 1
        return next_agent

    def next_response(self, observation):
        agent = self.next_agent()
        self.current_agent = agent
        model_response = self.current_agent.response(observation)
        self.dialog_history.append(model_response)
        return model_response

    def is_terminal(self):
        if self.current_agent.is_terminal():
            return True

    def get_rewards(self, ctxs):
        """Return the rewards of alice and bob.
        Returns:
            reward_1 (float): Reward of Alice.
            reward_2 (float): Reward of Bob.
        """
        choices = []
        for agent in [self.alice, self.bob]:
            choice = agent.choose()
            choices.append(choice)
        agree, rewards = self.alice.domain.score_choices(choices, ctxs)
        return agree, rewards

    def init_session(self):
        self.__turn_indicator = random.choice([0, 1])
        self.alice.init_session()
        self.bob.init_session()
