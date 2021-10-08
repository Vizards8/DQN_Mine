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

outputdir = hp.output_dir + '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.makedirs(outputdir, exist_ok=True)
writer = SummaryWriter(outputdir)


# 校验reward
def verify_reward(env):
    reward = 0
    for i in env.jobs:
        if not i.T_start:
            print(f'oops:job_id:{i.id} not start!')  # T_start未赋值，则打印错误
        elif i.T_start >= i.T_arrival and i.T_spent > 0:
            reward += i.T_deadline - i.T_start - i.T_spent
        else:
            # T_start/T_arrival/T_spent不满足条件，则打印错误
            print(f'job_id:{i.id}, T_arrival:{i.T_arrival}, T_start:{i.T_start}, T_spent:{i.T_spent}')
    return reward


def FIFO(env):
    env.reset()

    # 循环所有任务，因为是先来先服务从前往后顺序扫描即可
    machine_list = [[] for i in range(hp.machine_num)]
    for i in env.jobs:
        min_spent = []  # 计算每台机器的最早空闲时间+在这个机器上的花费时间
        for j in range(hp.machine_num):
            if machine_list[j] == []:  # 机器上没任务
                min_spent.append(i.T_arrival + env.machines[j].spent[i.type])
            else:  # 机器上有任务
                last_job = machine_list[j][-1]
                if i.T_arrival >= last_job.T_start + last_job.T_spent:
                    min_spent.append(i.T_arrival + env.machines[j].spent[i.type])
                else:
                    min_spent.append(last_job.T_start + last_job.T_spent + env.machines[j].spent[i.type])

        # 给不能做该任务的机器+10000s
        for j in range(hp.machine_num):
            if env.machines[j].spent[i.type] == -1:
                min_spent[j] += 10000

        # 决定在哪台机器上做
        id = min_spent.index(min(min_spent))
        if machine_list[id] == []:
            i.T_start = i.T_arrival
            i.T_spent = env.machines[id].spent[i.type]
        else:
            i.T_spent = env.machines[id].spent[i.type]
            last_job = machine_list[id][-1]
            if i.T_arrival >= last_job.T_start + last_job.T_spent:
                i.T_start = i.T_arrival
            else:
                i.T_start = last_job.T_start + last_job.T_spent
        machine_list[id].append(i)

    # 计算总延迟
    print(f'FIFO_reward:{verify_reward(env)}')
    return verify_reward(env)

    # # 错误的FIFO，就拿来作对比吧
    # env.reset()
    # # 计算每个type花费最少时间的机器
    # min_spent = []
    # for i in range(hp.type_num):
    #     temp = 100
    #     for j in env.machines:
    #         if j.spent[i] != -1 and j.spent[i] < temp:
    #             temp = j.spent[i]
    #             min_id = j.id
    #     min_spent.append(min_id)
    # print(f'花费时间最少的机器:{min_spent}')
    #
    # # 循环所以任务，因为是先来先服务从前往后顺序扫描即可
    # machine_list = [[] for i in range(hp.machine_num)]
    # for i in env.jobs:
    #     id = min_spent[i.type]  # 此任务在这个机器上做花费时间最短
    #     if machine_list[id] == []:  # 开始时这个机器为空的
    #         machine_list[id].append(i)
    #         i.T_start = i.T_arrival
    #         i.T_spent = env.machines[id].spent[i.type]
    #     else:
    #         i.T_spent = env.machines[id].spent[i.type]  # 花费时间不变，仍然在这个机器上做
    #         # 查找开始时间，上一个任务的结束时间/此任务到达时间
    #         last_job = machine_list[id][-1]
    #         if i.T_arrival >= last_job.T_start + last_job.T_spent:
    #             i.T_start = i.T_arrival
    #         else:
    #             i.T_start = last_job.T_start + last_job.T_spent
    #         machine_list[id].append(i)
    #
    # # 计算总延迟
    # print(f'wrong_FIFO_reward:{verify_reward(env)}')


def train(env, agent):
    print('Start to train !')
    print(f'Algorithm:{hp.algo}, Using Device:{device}')
    iteration = 0

    FIFO(env)
    # epoch
    for i_ep in range(hp.train_eps):
        env.reset()
        # env.re_random_job()
        ep_reward = 0
        reward_list = [0]
        loss_list = [0]

        # 按照时间循环
        job_cur_id = 0  # 定义当前(current)任务序号，不然需要全表扫描
        loop = tqdm(enumerate(np.arange(0, 1600, 0.01)), total=160000)
        for id, T in loop:
            iteration += 1
            # 提前终止
            if job_cur_id >= hp.job_num and len(env.waiting) == 0:
                if T >= env.jobs[hp.job_num - 1].T_arrival:
                    break
                else:
                    print('debug')

            # 打印用，防止有的loop没有reward
            reward = 0

            # debug用，错误打印
            if T >= 1599.99 and len(env.waiting) > 1:
                print('debug')

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
                        reward, feasible = env.step(action, env.jobs[job_cur_id], T)
                        reward_list.append(reward)
                        ep_reward += reward

                    # 有了可行的action，加入经验池
                    next_state = env.get_state(env.jobs[job_cur_id], T)
                    next_state[0] = 0
                    next_state[1] = 3
                    for spent in range(2, 2 + hp.machine_num):
                        next_state[spent] = -1
                    if job_cur_id >= env.job_num - 1 or T == 159.99:
                        done = True
                    else:
                        done = False
                    agent.memory.push(state, action, reward, next_state, done)
                    loss = agent.update()
                    if loss is not None:
                        # writer.add_scalar('train/Loss', loss, iteration)
                        loss_list.append(loss)
                    job_cur_id += 1

            # list中的任务按照priority排序
            env.waiting.sort(key=operator.attrgetter('priority', 'id'), reverse=False)

            # 更新所有机器剩余时间，取出完成的任务
            for i in env.machines:
                # 取出完成的任务
                if i.running:
                    if T >= (i.running.T_start + i.running.T_spent):
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

                            # 最后一个优化结算
                            if job_cur_id >= hp.job_num and env.waiting == []:
                                flag = False
                                for j in range(hp.machine_num):
                                    if cur_mask[j] == 1:
                                        flag = True
                                        break
                                if flag:
                                    for j in range(hp.machine_num, hp.action_dim):
                                        cur_mask[j] = 0

                            action = agent.choose_action(state, cur_mask)
                            reward, feasible = env.step(action, job_prime, T)
                            reward_list.append(reward)
                            ep_reward += reward
                        # 有了可行的action，加入经验池
                        # next_state = env.get_state(env.jobs[job_cur_id], T)
                        next_state[0] = 0
                        next_state[1] = 3
                        for spent in range(2, 2 + hp.machine_num):
                            next_state[spent] = -1
                        if job_cur_id >= env.job_num:
                            done = True
                        else:
                            done = False
                        agent.memory.push(state, action, reward, next_state, done)
                        loss = agent.update()
                        if loss is not None:
                            # writer.add_scalar('train/Loss', loss, iteration)
                            loss_list.append(loss)

            # log
            writer.add_scalar('train/Reward', reward_list[-1], iteration)
            writer.add_scalar('train/Loss', loss_list[-1], iteration)
            loop.set_description(f'Epoch_Train [{i_ep + 1}/{hp.train_eps}]')
            loop.set_postfix({
                'reward': '{0:1.5f}'.format(reward),
                'ep_reward': '{0:1.5f}'.format(ep_reward)
            })

        # verify
        if round(verify_reward(env), 5) == round(ep_reward, 5):
            print('Verified!')
        else:
            print(f'verify_reward:{verify_reward(env)}, ep_reward:{ep_reward}')

        # epoch log
        writer.add_scalar('train/epoch_reward', ep_reward, i_ep)

        if (i_ep + 1) % hp.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    writer.close()
    print('Complete training！')


def eval(env, agent):
    print('Start to eval !')
    print(f'Algorithm:{hp.algo}, Using Device:{device}')
    writer = SummaryWriter(outputdir)
    ma_rewards = []  # moving average rewards
    iteration = 0

    # epoch
    for i_ep in range(hp.eval_eps):
        env.re_random_job()
        FIFO(env)
        env.reset()
        ep_reward = 0
        reward_list = [0]

        # 按照时间循环
        job_cur_id = 0  # 定义当前(current)任务序号，不然需要全表扫描
        loop = tqdm(enumerate(np.arange(0, 1600, 0.01)), total=160000)
        for id, T in loop:
            iteration += 1
            # 提前终止
            if job_cur_id >= hp.job_num and len(env.waiting) == 0:
                if T >= env.jobs[hp.job_num - 1].T_arrival:
                    break
                else:
                    print('debug')

            # 打印用，防止有的loop没有reward
            reward = 0

            # debug用，错误打印
            if T >= 1599.99 and len(env.waiting) > 1:
                print('debug')

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
                        reward, feasible = env.step(action, env.jobs[job_cur_id], T)
                        reward_list.append(reward)
                        ep_reward += reward

                    job_cur_id += 1

            # list中的任务按照priority排序
            env.waiting.sort(key=operator.attrgetter('priority', 'id'), reverse=False)

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

                            # 最后一个优化结算
                            if job_cur_id >= hp.job_num and env.waiting == []:
                                flag = False
                                for j in range(hp.machine_num):
                                    if cur_mask[j] == 1:
                                        flag = True
                                        break
                                if flag:
                                    for j in range(hp.machine_num, hp.action_dim):
                                        cur_mask[j] = 0

                            action = agent.choose_action(state, cur_mask)
                            reward, feasible = env.step(action, job_prime, T)
                            reward_list.append(reward)
                            ep_reward += reward

            # log
            writer.add_scalar('valid/Reward', reward_list[-1], iteration)
            loop.set_description(f'Epoch_Valid [{i_ep + 1}/{hp.eval_eps}]')
            loop.set_postfix({
                'reward': '{0:1.5f}'.format(reward),
                'ep_reward': '{0:1.5f}'.format(ep_reward)
            })

        # verify
        if round(verify_reward(env), 5) == round(ep_reward, 5):
            print('Verified!')
        else:
            print(f'verify_reward:{verify_reward(env)}, ep_reward:{ep_reward}')

        # epoch log
        writer.add_scalar('valid/epoch_reward', ep_reward, i_ep)

        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        # if (i_ep + 1) % 10 == 10:
        #     print(f"Episode:{i_ep + 1}/{hp.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return ma_rewards


if __name__ == "__main__":
    # train
    agent = DQN()
    env = SchedulingEnv(hp.job_num, hp.machine_num, hp.type_num)
    while True:
        env.init()
        if FIFO(env) > 0 and FIFO(env) < 2600:
            break
    # train(env, agent)
    # os.makedirs(hp.output_dir, exist_ok=True)
    # agent.save(path=hp.model_path)

    # eval
    agent = DQN()
    agent.load(path=hp.model_path)
    ma_rewards = eval(env, agent)
