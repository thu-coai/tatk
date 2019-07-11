import sys
import os
from tatk.dialog_agent import PipelineAgent, BiSession
from tatk.nlu.svm.multiwoz import SVMNLU
from tatk.nlu.bert.multiwoz import BERTNLU
from tatk.dst.rule.multiwoz import RuleDST
from tatk.policy.rule.multiwoz import Rule
from tatk.nlg.template_nlg.multiwoz import TemplateNLG

# svm nlu trained on usr sentence of multiwoz
nlu = SVMNLU('usr',model_file="https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip")
# simple rule DST
dst = RuleDST()
# rule policy
policy = Rule(character='sys')
# template NLG
nlg = TemplateNLG(is_user=False)
# assemble
sys_agent = PipelineAgent(nlu, dst, policy, nlg)

# svm nlu trained on sys sentence of multiwoz
nlu = BERTNLU('sys',model_file="https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_sys.zip")
# not use dst
dst = None
# rule policy
policy = Rule(character='usr')
# template NLG
nlg = TemplateNLG(is_user=True)
# assemble
simulator = PipelineAgent(nlu, dst, policy, nlg)

sess = BiSession(sys_agent, simulator, None)

sys_response = 'null'
sess.init_session()
for i in range(30):
    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    print('user:', user_response)
    print('sys:', sys_response)
    print('reward:', reward)
    if session_over is True:
        break