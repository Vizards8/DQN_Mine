from random import random, uniform, randint

from agent import DQN
from hparam import hparams as hp
import numpy as np


class job:
    def __init__(self, last_arrival, id):
        self.id = id
        self.T_arrival = last_arrival + uniform(0, 1)  # 到达时间
        self.T_deadline = self.T_arrival + uniform(3, 5)  # 截止
        self.type = randint(0, hp.type_num - 1)  # 任务类别
        self.mask = np.ones(hp.action_dim)  # mask
        self.T_spent = None  # 耗时
        self.T_start = None  # 开始时间
        self.priority = 2  # 优先度

    def reset(self):
        self.T_spent = None
        self.T_start = None
        self.priority = 2


class machine:
    def __init__(self, id):
        self.id = id
        self.spent = []  # list:每个type在这个机器上运行的时间
        self.running = None

    def reset(self):
        self.running = None


class SchedulingEnv:
    def __init__(self, job_num, machine_num, type_num):
        self.job_num = job_num
        self.machine_num = machine_num  # 机器个数
        self.type_num = type_num  # 任务type个数
        self.machines = [machine(i) for i in range(self.machine_num)]
        self.jobs = []
        self.waiting = []  # 一个waiting_list

    def reset(self):
        for i in self.jobs:
            i.reset()
        for i in self.machines:
            i.reset()
        self.waiting = []

    def re_random_job(self):
        # 重新随机job
        self.jobs = []
        # 随机第一个job
        data = job(0, 0)
        for j in self.machines:
            if j.spent[data.type] == -1:
                data.mask[j.id] = 0
        self.jobs.append(data)
        # 随机剩余的job
        for i in range(1, self.job_num):
            data = job(self.jobs[-1].T_arrival, i)
            for j in self.machines:
                if j.spent[data.type] == -1:
                    data.mask[j.id] = 0
            self.jobs.append(data)

    # 初始化数据
    def init(self):
        print('Initializing data...')
        self.reset()

        # # 随机每个machine不同type的service时间
        # re_random = True  # True:要重新随机
        # while re_random:
        #     for i in self.machines:
        #         # 防止一台机器什么任务也做不了
        #         machine_flag = True
        #         while machine_flag:
        #             i.spent = []
        #             for j in range(self.type_num):
        #                 if random() > 0.5:  # 随机数判断在这个机器上能不能做该任务
        #                     i.spent.append(uniform(2.5, 5.5))
        #                 else:
        #                     i.spent.append(-1)
        #             for k in i.spent:
        #                 if not k == -1:
        #                     machine_flag = False
        #         print('machine_id:', i.id, i.spent)
        #     # 防止一个任务什么机器都做不了
        #     for ii in range(self.type_num):
        #         print('我执行了吗')
        #         pass_id = False
        #         for jj in self.machines:
        #             if not jj.spent[ii] == -1:
        #                 pass_id = True
        #                 break
        #         if not pass_id:
        #             re_random = False
        #             break

        # 随机每个machine不同type的service时间
        if self.type_num == 10 and self.machine_num == 30:
            self.machines[0].spent = [-1, -1, -1, -1, -1, 4.1, 1.8, 1.4, 3.4, -1]
            self.machines[1].spent = [-1, -1, -1, -1, -1, 2.9, -1, 1.6, -1, 3.9]
            self.machines[2].spent = [-1, 4.8, -1, 3.4, -1, -1, -1, -1, -1, 4.7]
            self.machines[3].spent = [-1, -1, -1, -1, -1, 1.4, 2.4, 1.0, 1.6, -1]
            self.machines[4].spent = [1.2, 3.3, -1, -1, 3.1, 2.3, -1, -1, -1, -1]
            self.machines[5].spent = [4.2, 1.8, -1, -1, 3.3, 1.1, -1, -1, -1, -1]
            self.machines[6].spent = [4.0, 1.0, -1, -1, 4.6, 4.2, -1, -1, -1, -1]
            self.machines[7].spent = [4.9, 2.7, 4.2, 2.2, -1, -1, -1, -1, -1, -1]
            self.machines[8].spent = [-1, -1, -1, -1, -1, 1.5, 4.3, 4.8, 4.9, -1]
            self.machines[9].spent = [2.6, 4.3, -1, -1, 1.8, 5.0, -1, -1, -1, -1]
            self.machines[10].spent = [-1, -1, -1, -1, -1, -1, -1, 4.8, 4.1, 3.6]
            self.machines[11].spent = [-1, -1, -1, -1, -1, 1.8, -1, 2.6, -1, 4.6]
            self.machines[12].spent = [-1, -1, -1, -1, -1, 2.1, -1, 2.4, -1, 3.7]
            self.machines[13].spent = [-1, -1, -1, 3.0, -1, 3.7, -1, -1, -1, 2.8]
            self.machines[14].spent = [-1, 2.2, -1, 3.1, -1, -1, -1, -1, -1, 3.3]
            self.machines[15].spent = [-1, -1, -1, -1, -1, 4.6, -1, 3.7, -1, 3.2]
            self.machines[16].spent = [4.5, 1.4, 4.2, 1.5, -1, -1, -1, -1, -1, -1]
            self.machines[17].spent = [4.8, -1, 1.1, -1, 2.3, -1, -1, -1, -1, -1]
            self.machines[18].spent = [2.0, 3.0, 2.2, 1.1, -1, -1, -1, -1, -1, -1]
            self.machines[19].spent = [-1, -1, -1, -1, -1, 2.1, 1.9, 4.3, 3.0, -1]
            self.machines[20].spent = [-1, -1, -1, -1, -1, -1, -1, 1.9, 3.6, 3.8]
            self.machines[21].spent = [2.2, -1, 3.3, -1, 3.6, -1, -1, -1, -1, -1]
            self.machines[22].spent = [-1, -1, -1, 4.3, -1, 4.0, -1, -1, -1, 3.4]
            self.machines[23].spent = [-1, -1, -1, -1, -1, -1, -1, 2.3, 4.6, 1.2]
            self.machines[24].spent = [3.9, -1, 1.1, -1, 4.2, -1, -1, -1, -1, -1]
            self.machines[25].spent = [-1, -1, -1, 2.2, -1, 1.9, -1, -1, -1, 1.8]
            self.machines[26].spent = [-1, 2.0, -1, 3.8, -1, -1, -1, -1, -1, 1.6]
            self.machines[27].spent = [-1, -1, -1, -1, -1, -1, -1, 4.4, 2.4, 4.3]
            self.machines[28].spent = [2.8, -1, 2.4, -1, 1.7, -1, -1, -1, -1, -1]
            self.machines[29].spent = [2.2, 4.0, 1.3, 4.9, -1, -1, -1, -1, -1, -1]
            for i in self.machines:
                print('machine_id:', i.id, i.spent)
        else:
            for i in self.machines:
                i.spent = []
                for j in range(self.type_num):
                    if random() > 0.4:  # 随机数判断在这个机器上能不能做该任务
                        i.spent.append(uniform(2.5, 5.5))
                    else:
                        i.spent.append(-1)
                print('machine_id:', i.id, i.spent)

        # 随机job
        self.re_random_job()

    def get_state(self, job, T):
        state = [T - job.T_arrival,  # t_suspended
                 job.priority,
                 len(self.waiting)  # list.number
                 ]

        for i in self.machines:
            # t_spent
            state.append(i.spent[job.type])
            # t_balance
            state.append(job.T_deadline - T - i.spent[job.type])
            # machine_time_let:机器上正在做的任务的剩余时间
            if i.running:
                state.append(i.running.T_start + i.running.T_spent - T)
            else:
                state.append(0)
            # machine_balance:机器上正在做的任务如果完成了的时间结余
            if i.running:
                state.append(i.running.T_deadline - T - i.spent[i.running.type])
            else:
                state.append(0)

        return state

    def step(self, action, job, T):
        reward = 0
        feasible = True

        # 添加进waiting_list
        if action >= self.machine_num:
            job.priority = (action - 5) / (hp.action_dim - self.machine_num - 1)
            self.waiting.append(job)
            # reward -= max(0, T - 150)

        else:
            # 所需执行时间是负数
            if self.machines[action].spent[job.type] < 0:
                reward += -0.001
                feasible = False

            # 机器的剩余任务时间是正
            if self.machines[action].running:
                reward += -0.001
                feasible = False

            # 可行，安排进机器
            if reward == 0:
                job.T_start = T
                job.T_spent = self.machines[action].spent[job.type]
                self.machines[action].running = job
                reward = job.T_deadline - T - job.T_spent

        # print(f'reward: {reward}')
        return reward, feasible

# # For the test
# machines = [[] for i in range(5)]
# print(machines)
# data = pd.read_excel('./data.xlsx')
# for i in data:
#     print(i)
