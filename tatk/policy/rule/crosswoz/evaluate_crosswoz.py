from tatk.policy.mle.crosswoz.mle import MLE
from tatk.policy.ppo.ppo import PPO
from tatk.dst.rule.crosswoz.state_tracker import RuleDST
from tatk.util.crosswoz.state import default_state
from tatk.policy.rule.crosswoz.rule_simulator import Simulator
from tatk.dialog_agent import PipelineAgent, BiSession
from tatk.util.crosswoz.lexicalize import delexicalize_da
import os
import zipfile
import json
from copy import deepcopy
import random
import numpy as np
from pprint import pprint
import torch


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for quad in predicts:
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if quad not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, F1


def evaluate_corpus_f1(policy, data, goal_type=None):
    dst = RuleDST()
    da_predict_golden = []
    delex_da_predict_golden = []
    for task_id, sess in data.items():
        if goal_type and sess['type']!=goal_type:
            continue
        dst.init_session()
        for i, turn in enumerate(sess['messages']):
            if turn['role'] == 'usr':
                dst.update(usr_da=turn['dialog_act'])
                if i + 2 == len(sess):
                    dst.state['terminated'] = True
            else:
                for domain, svs in turn['sys_state'].items():
                    for slot, value in svs.items():
                        if slot != 'selectedResults':
                            dst.state['belief_state'][domain][slot] = value
                golden_da = turn['dialog_act']

                predict_da = policy.predict(deepcopy(dst.state))
                # print(golden_da)
                # print(predict_da)
                # print()
                da_predict_golden.append({
                    'predict': predict_da,
                    'golden': golden_da
                })
                delex_da_predict_golden.append({
                    'predict': delexicalize_da(predict_da),
                    'golden': delexicalize_da(golden_da)
                })
                # print(delex_da_predict_golden[-1])
                dst.state['system_action'] = golden_da
        # break
    print('origin precision/recall/f1:',calculateF1(da_predict_golden))
    print('delex precision/recall/f1:', calculateF1(delex_da_predict_golden))


def evaluate_simulation(policy):
    usr_policy = Simulator()
    usr_agent = PipelineAgent(None, None, usr_policy, None)
    sys_policy = policy
    sys_dst = RuleDST()
    sys_agent = PipelineAgent(None, sys_dst, sys_policy, None)
    sess = BiSession(sys_agent=sys_agent, user_agent=usr_agent)

    # random_seed = 2019
    # random.seed(random_seed)
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)

    task_success = {'All': list(), '单领域': list(), '独立多领域': list(), '独立多领域+交通': list(), '不独立多领域': list(),
                    '不独立多领域+交通': list()}
    simulate_sess_num = 100
    repeat = 5
    while True:
        sys_response = []
        sess.init_session()
        # print(usr_policy.goal_type)
        if len(task_success[usr_policy.goal_type]) == simulate_sess_num*repeat:
            continue
        for i in range(20):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            # print('user:', user_response)
            # print('sys:', sys_response)
            # print(session_over, reward)
            # print()
            if session_over is True:
                task_success['All'].append(1)
                task_success[usr_policy.goal_type].append(1)
                break
        else:
            task_success['All'].append(0)
            task_success[usr_policy.goal_type].append(0)
        print([len(x) for x in task_success.values()])
        # print(min([len(x) for x in task_success.values()]))
        if min([len(x) for x in task_success.values()]) == simulate_sess_num*repeat:
            break
        # pprint(usr_policy.original_goal)
        # pprint(task_success)
    print('task_success')
    for k, v in task_success.items():
        print(k)
        for i in range(repeat):
            samples = v[i*simulate_sess_num:(i+1)*simulate_sess_num]
            print(sum(samples),len(samples),sum(samples)/len(samples))


if __name__ == '__main__':
    random_seed = 2019
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    test_data = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../data/crosswoz/final_test.json.zip'))
    test_data = read_zipped_json(test_data, 'final_test.json')
    policy = MLE()
    # policy = PPO(dataset='CrossWOZ', is_train=True,pretrain_model_path='../../../tatk/policy/mle/crosswoz/save/best')
    # policy = PPO(dataset='CrossWOZ')
    for goal_type in ['单领域','独立多领域','独立多领域+交通','不独立多领域','不独立多领域+交通',None]:
        print(goal_type)
        evaluate_corpus_f1(policy, test_data, goal_type=goal_type)
    evaluate_simulation(policy)
