import os, time
import torch
from agent import DQN
from hparam import hparams as hp
from scheduling_env import *
import operator
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, agent):
    print('Start to train !')
    print(f'Algorithm:{hp.algo}, Using Device:{device}')
    outputdir = hp.output_dir + '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(outputdir, exist_ok=True)
    writer = SummaryWriter(outputdir)
    iteration = 0

    env.init()
    # epoch
    for i_ep in range(hp.train_eps):
        env.reset()
        ep_reward = 0
        reward_list = [0]

        # 按照时间循环
        job_cur_id = 0  # 定义当前(current)任务序号，不然需要全表扫描
        loop = tqdm(enumerate(np.arange(0, 160, 0.01)), total=16000)
        for id, T in loop:
            iteration += 1

            # 新任务来了
            if job_cur_id < env.job_num:
                if T >= env.jobs[job_cur_id].T_arrival:
                    state = env.get_state(env.jobs[job_cur_id], T)
                    feasible = False
                    while not feasible:
                        # 记录当前mask
                        cur_mask = env.jobs[job_cur_id].mask * np.ones(hp.action_dim)
                        for machine in env.machines:
                            if machine.running:
                                cur_mask[machine.id] = 0
                        action = agent.choose_action(state, cur_mask)
                        # print(f'action:{action}')
                        reward, feasible = env.step(action, env.jobs[job_cur_id], T)
                        reward_list.append(reward)
                        ep_reward += reward

                    # 有了可行的action，加入经验池
                    next_state = env.get_state(env.jobs[job_cur_id], T)
                    if job_cur_id >= env.job_num - 1 or T == 159.99:
                        done = True
                    else:
                        done = False
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.update()
                    job_cur_id += 1

            # list中的任务按照priority排序
            env.waiting.sort(key=operator.attrgetter('priority'), reverse=False)

            # 更新所有机器剩余时间，取出完成的任务
            for i in env.machines:
                # 取出完成的任务
                if i.running:
                    if (i.running.T_start + i.running.T_spent) >= T:
                        i.running = None
                # 取出最优先任务
                if not i.running:
                    if len(env.waiting) > 0:
                        job_prime = env.waiting.pop(0)
                        state = env.get_state(job_prime, T)

                        feasible = False
                        while not feasible:
                            # 记录当前mask
                            cur_mask = job_prime.mask * np.ones(hp.action_dim)
                            for machine in env.machines:
                                if machine.running:
                                    cur_mask[machine.id] = 0
                            if job_prime.mask[0] == job_prime.mask[1] == job_prime.mask[2] == job_prime.mask[3] == \
                                    job_prime.mask[4] == 0:
                                print('debug')
                            action = agent.choose_action(state, cur_mask)
                            # print(f'action:{action}')
                            reward, feasible = env.step(action, job_prime, T)
                            reward_list.append(reward)
                            ep_reward += reward
                        # 有了可行的action，加入经验池
                        next_state = env.get_state(job_prime, T)
                        if job_cur_id >= env.job_num:
                            done = True
                        else:
                            done = False
                        agent.memory.push(state, action, reward, next_state, done)
                        agent.update()

            # log
            writer.add_scalar('train/last_reward', reward_list[-1], iteration)
            loop.set_description(f'Epoch_Train [{i_ep + 1}/{hp.train_eps}]')
            loop.set_postfix({
                'last_reward': '{0:1.5f}'.format(reward_list[-1]),
                'ep_reward': '{0:1.5f}'.format(ep_reward)
            })

        # epoch log
        writer.add_scalar('train/epoch_reward', ep_reward, i_ep)

        if (i_ep + 1) % hp.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    writer.close()
    print('Complete training！')


def eval(env, agent):
    print('Start to eval !')
    print(f'Algorithm:{hp.algo}, Using Device:{device}')
    outputdir = hp.output_dir + '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(outputdir, exist_ok=True)
    writer = SummaryWriter(outputdir)
    rewards = []
    ma_rewards = []  # moving average rewards
    iteration = 0

    # epoch
    for i_ep in range(hp.eval_eps):
        env.init()
        env.reset()
        ep_reward = 0
        reward_list = [0]

        # 按照时间循环
        job_cur_id = 0  # 定义当前(current)任务序号，不然需要全表扫描
        loop = tqdm(enumerate(np.arange(0, 160, 0.01)), total=16000)
        for id, T in loop:
            if T >= 159.99 and len(env.waiting) > 1:
                print('debug')
            iteration += 1

            # 新任务来了
            if job_cur_id < env.job_num:
                if T >= env.jobs[job_cur_id].T_arrival:
                    state = env.get_state(env.jobs[job_cur_id], T)
                    feasible = False
                    while not feasible:
                        # 记录当前mask
                        cur_mask = env.jobs[job_cur_id].mask * np.ones(hp.action_dim)
                        for machine in env.machines:
                            if machine.running:
                                cur_mask[machine.id] = 0
                        action = agent.choose_action(state, cur_mask)
                        # print(f'action:{action}')
                        reward, feasible = env.step(action, env.jobs[job_cur_id], T)
                        reward_list.append(reward)
                        ep_reward += reward

                    job_cur_id += 1

            # list中的任务按照priority排序
            env.waiting.sort(key=operator.attrgetter('priority'), reverse=False)

            # 更新所有机器剩余时间，取出完成的任务
            for i in env.machines:
                # 取出完成的任务
                if i.running:
                    if (i.running.T_start + i.running.T_spent) >= T:
                        i.running = None
                # 取出最优先任务
                if not i.running:
                    if len(env.waiting) > 0:
                        job_prime = env.waiting.pop(0)
                        state = env.get_state(job_prime, T)

                        feasible = False
                        while not feasible:
                            # 记录当前mask
                            cur_mask = job_prime.mask * np.ones(hp.action_dim)
                            for machine in env.machines:
                                if machine.running:
                                    cur_mask[machine.id] = 0
                            action = agent.choose_action(state, cur_mask)
                            # print(f'action:{action}')
                            reward, feasible = env.step(action, job_prime, T)
                            reward_list.append(reward)
                            ep_reward += reward

            # log
            writer.add_scalar('valid/last_reward', reward_list[-1], iteration)
            loop.set_description(f'Epoch_Valid [{i_ep + 1}/{hp.eval_eps}]')
            loop.set_postfix({
                'last_reward': '{0:1.5f}'.format(reward_list[-1]),
                'ep_reward': '{0:1.5f}'.format(ep_reward)
            })

        # epoch log
        writer.add_scalar('valid/epoch_reward', ep_reward, i_ep)

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        # if (i_ep + 1) % 10 == 10:
        #     print(f"Episode:{i_ep + 1}/{hp.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return rewards, ma_rewards


if __name__ == "__main__":
    # train
    agent = DQN()
    env = SchedulingEnv(hp.job_num, hp.machine_num, hp.type_num)
    rewards = train(env, agent)
    os.makedirs(hp.output_dir, exist_ok=True)
    agent.save(path=hp.model_path)

    # eval
    agent = DQN()
    env = SchedulingEnv(hp.job_num, hp.machine_num, hp.type_num)
    agent.load(path=hp.model_path)
    rewards, ma_rewards = eval(env, agent)
