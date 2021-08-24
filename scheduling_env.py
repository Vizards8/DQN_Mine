from random import random, uniform, randint


class job:
    def __init__(self, last_arrival, id):
        self.id = id
        self.T_arrival = last_arrival + uniform(0.5, 1.5)  # 到达时间
        self.T_service = None  # 耗时
        self.T_deadline = self.T_arrival + uniform(3, 5)  # 截止
        self.type = randint(0, 2)  # 任务类别
        self.T_start = None  # 开始时间
        self.done = False
        self.Response_rate = None  # 响应比 = (1 + T拖延)/T给定

    def reset(self):
        self.T_service = None
        self.T_start = None
        self.Response_rate = None

    def cal_response(self, T):
        if T > self.T_arrival:
            delay = T - self.T_arrival
            given = self.T_deadline - T
            self.Response_rate = (1 + delay) / given


class machine:
    def __init__(self, id):
        self.id = id
        self.service = []  # 每个type在这个机器上运行的时间
        self.running = None
        self.waiting = []

    def add_running(self, T):
        self.running = None
        if len(self.waiting):
            self.running = self.waiting.pop(0)  # 取出下一个任务，放到执行中
            self.running.T_start = T

    def add_waiting(self, job, T):
        # 记录T_start
        if self.running:
            if len(self.waiting):
                last_job = self.waiting[-1]
                job.T_start = last_job.T_start + last_job.T_service
            else:
                last_job = self.running
                job.T_start = last_job.T_start + last_job.T_service
        else:
            job.T_start = T
        # 记录T_service
        job.T_service = self.service[job.type]
        # 添加进waiting
        self.waiting.append(job)


class SchedulingEnv:
    def __init__(self, job_num, service_num, type_num):
        self.job_num = job_num
        self.service_num = service_num
        self.type_num = type_num
        self.machines = [machine(i) for i in range(self.service_num)]
        self.jobs = []
        self.type2service = []

    # 重新随机数据
    def reset(self):
        self.machines = [machine(i) for i in range(self.service_num)]
        self.jobs = []
        self.type2service = []

        # 随机job
        self.jobs.append(job(0, 0))
        for i in range(1, self.job_num):
            data = job(self.jobs[-1].T_arrival, i)
            self.jobs.append(data)

        # 随机每个machine不同type的service时间
        for i in self.machines:
            for j in range(self.service_num):
                if random() > 0.5:
                    i.service.append(uniform(2.5, 5.5))
                else:
                    i.service.append(1000)
            # print('type:', i.id, i.service)

    def get_state(self, job, T):
        state = [T,  # 环境时间
                 job.T_arrival,
                 # job.T_service,
                 job.T_deadline,
                 job.type
                 ]
        for i in self.machines:
            if i.running:
                t_end_now = i.running.T_start + i.running.T_service  # 当前任务结束时间
            else:
                t_end_now = 0
            state.append(t_end_now)
            if len(i.waiting):
                t_end_last = i.waiting[-1].T_start + i.waiting[-1].T_service  # 最后一个任务结束时间
            else:
                t_end_last = 0
            state.append(t_end_last)
        return state

    def step(self, action, job, T):
        # 添加进对应的machine.waiting
        self.machines[action].add_waiting(job, T)

        # 刷新机器，如果必要
        for i in self.machines:
            # running有任务且完成
            if i.running and i.running.T_start + i.running.T_service <= T:
                self.jobs[i.running.id].done = True
                i.add_running(T)
            # running压根没任务
            if i.running is None:
                i.add_running(T)

        # 计算reward，超时-1，未超时+1
        # 分类计算，done，running，waiting
        reward = 0
        for i in self.jobs:
            if i.done:
                reward += self.cal_tardiness(i)
        for i in self.machines:
            if i.running:
                reward += self.cal_tardiness(i.running)
            if len(i.waiting):
                for j in i.waiting:
                    reward += self.cal_tardiness(j)

        return reward

    def cal_tardiness(self, job):
        # 计算超时的时间
        tardiness = job.T_start + job.T_service - job.T_deadline
        reward = -tardiness if tardiness >= 0 else tardiness
        return reward

# # For the test
# machines = [[] for i in range(5)]
# print(machines)
# data = pd.read_excel('./data.xlsx')
# for i in data:
#     print(i)
