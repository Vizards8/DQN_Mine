import os, time
import torch
from agent import DQN
from hparam import hparams as hp
from scheduling_env import *
import operator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, agent):
    print('Start to train !')
    print(f'Algorithm:{hp.algo}, Using Device:{device}')
    os.makedirs(hp.output_dir, exist_ok=True)
    writer = SummaryWriter(hp.output_dir)
    iteration = 0

    # epoch
    for i_ep in range(hp.train_eps):
        env.reset()
        start_time = time.time()
        ep_reward = 0
        done = False

        # 任务一个个来
        loop = tqdm(enumerate(env.jobs), total=len(env.jobs))
        for id, job_curr in loop:
            iteration += 1
            while time.time() - start_time < job_curr.T_arrival:
                pass

            # 冻结时间
            T = time.time() - start_time
            # print(f'job arrived!T:{T}, id:{job_curr.id}, T_arrival:{job_curr.T_arrival}')

            # 取出所有机器的waiting队列 + Job
            # 清空所有机器的waiting队列
            job_reschedul = [job_curr]
            for machine in env.machines:
                job_reschedul += machine.waiting
                machine.waiting = []

            # 计算响应比
            for job in job_reschedul:
                job.cal_response(T)
            job_reschedul.sort(key=operator.attrgetter('Response_rate'), reverse=True)

            # 开始重排
            reward = 0
            for id_reschedul, job in enumerate(job_reschedul):
                state = env.get_state(job, T)
                action = agent.choose_action(state)
                reward = env.step(action, job, T)
                if id_reschedul < len(job_reschedul) - 1:
                    next_state = env.get_state(job_reschedul[id_reschedul + 1], T)
                else:
                    if id < len(env.jobs) - 1:
                        next_state = env.get_state(env.jobs[id + 1], T)
                    else:
                        next_state = [0 for i in range(hp.state_dim)]
                        done = True
                reward += reward

            # 当前任务重排完毕
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()

            # log
            writer.add_scalar('Reward', reward, iteration)
            loop.set_description(f'Epoch_Train [{i_ep}/{hp.train_eps}]')
            loop.set_postfix({
                'reward': '{0:1.5f}'.format(reward),
                'mean_reward': '{0:1.5f}'.format(ep_reward/len(env.jobs))
            })

        if (i_ep + 1) % hp.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    writer.close()
    print('Complete training！')


# def eval(env, agent):
#     print('Start to eval !')
#     print(f'Env:{hp.env}, Algorithm:{hp.algo}, Device:{device}')
#     rewards = []
#     ma_rewards = []  # moving average rewards
#     for i_ep in range(hp.eval_eps):
#         ep_reward = 0  # reward per episode
#         state = env.reset()
#         while True:
#             action = agent.predict(state)
#             next_state, reward, done, _ = env.step(action)
#             state = next_state
#             ep_reward += reward
#             if done:
#                 break
#         rewards.append(ep_reward)
#         if ma_rewards:
#             ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
#         else:
#             ma_rewards.append(ep_reward)
#         if (i_ep + 1) % 10 == 10:
#             print(f"Episode:{i_ep + 1}/{hp.eval_eps}, reward:{ep_reward:.1f}")
#     print('Complete evaling！')
#     return rewards, ma_rewards


if __name__ == "__main__":
    # train
    agent = DQN()
    env = SchedulingEnv(hp.job_num, hp.service_num, hp.type_num)
    rewards = train(env, agent)
    os.makedirs(hp.output_dir, exist_ok=True)
    agent.save(path=hp.model_path)

    # eval
    # env, agent = env_agent_config(hp, seed=10)
    # agent.load(path=hp.model_path)
    # rewards, ma_rewards = eval(hp, env, agent)
