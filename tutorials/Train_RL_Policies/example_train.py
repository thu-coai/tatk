# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019

@author: truthless
"""

import numpy as np
import torch
from torch import multiprocessing as mp
from tatk.dialog_agent.agent import PipelineAgent
from tatk.dialog_agent.env import Environment
from tatk.nlu.svm.multiwoz import SVMNLU
from tatk.dst.rule.multiwoz import RuleDST
from tatk.policy.rule.multiwoz import Rule
from tatk.policy.ppo import PPO
from tatk.policy.rlmodule import Memory, Transition
from tatk.nlg.template_nlg.multiwoz import TemplateNLG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s).cpu()

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1
            
            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))
            
            # save to queue
            buff.push(s_vec.numpy(), a.numpy(), r, mask, next_s_vec.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()

def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_) # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()

def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)
    
    policy.update(epoch, batchsz_real, s, a, r, mask)    

if __name__ == '__main__':
    # svm nlu trained on usr sentence of multiwoz
    nlu_sys = SVMNLU('usr', model_file="https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip")
    # simple rule DST
    dst_sys = RuleDST()
    # rule policy
    policy_sys = PPO(True)
    # template NLG
    nlg_sys = TemplateNLG(is_user=False)    
    
    # svm nlu trained on sys sentence of multiwoz
    nlu_usr = SVMNLU('sys', model_file="https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_sys.zip")
    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = Rule(character='usr')
    # template NLG
    nlg_usr = TemplateNLG(is_user=True)
    # assemble
    simulator = PipelineAgent(nlu_usr, dst_usr, policy_usr, nlg_usr)
    
    env = Environment(nlg_sys, simulator, nlu_sys, dst_sys)
    
    batchsz = 1024
    epoch = 20
    process_num = 8
    update(env, policy_sys, batchsz, epoch, process_num)

