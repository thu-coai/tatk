from tatk.nlu import NLU
from tatk.dst import Tracker
from tatk.policy import Policy
from tatk.nlg import NLG
from tatk.dialog_agent import Agent, PipelineAgent
from tatk.dialog_agent import Session, BiSession, DealornotSession

from os.path import abspath, dirname


def get_root_path():
    return dirname(dirname(abspath(__file__)))
