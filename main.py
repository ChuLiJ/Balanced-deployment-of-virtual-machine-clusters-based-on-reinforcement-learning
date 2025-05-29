import torch
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from env import generate_pms, generate_vms, cluster_vms, VirtualMachineClusterEnv, train
from DQN import DQN, ReplayBuffer, DoubleDQN, DDDQN, PrioritizedReplayBuffer, NStepBuffer, RainbowDQN
from PPO import PPO
from reward import Reward
from utils import print_distribution, moving_average, print_returns, print_utilization, sample_from_buffer
from contrast import greedy_migration
from config import RainbowConfig

# 设定超参数
lr = 1e-3
actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 1000
num_per_episodes = 100
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2

epsilon = 0.5
decay = 0.995
min_epsilon = 0.01

# time_array = np.zeros(10)
# reward_array = np.zeros(10)
# indictor_array = np.zeros(10)
# for i in range(10):
#     time, reward, indictor = train(gamma, lr, actor_lr, critic_lr, epsilon, lmbda, epochs, eps, num_episodes,
#                                    num_per_episodes)
#     time_array[i] = time
#     reward_array[i] = reward
#     indictor_array[i] = indictor
#
# time_mean = time_array.mean()
# reward_mean = reward_array.mean()
# indictor_mean = indictor_array.mean()
#
# print("Time：", time_mean)
# print("Reward：", reward_mean)
# print("HRL Reward：", indictor_mean)


if not RainbowConfig.use_multi_step:
    n_step = 1
else:
    n_step = 3
target_update = 10
buffer_size = 10000
minimal_size = 100
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_pms = 100
num_vms = 400
pm_cpu_range = (32, 128)
pm_mem_range = (128, 512)

pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
vms = generate_vms(num_vms)
vms, centers = cluster_vms(vms, 3)

env = VirtualMachineClusterEnv(pms, vms)
env_name = "VirtualMachineCluster"

if RainbowConfig.use_prioritized_replay:
    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
else:
    replay_buffer = ReplayBuffer(capacity=buffer_size)
n_step_buffer = NStepBuffer(n_step, gamma=gamma)

state_dim = len(env.get_state(env.develop))
action_dim = 5

if RainbowConfig.use_distributional_rl:
    agent_dqn = RainbowDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
else:
    agent_dqn = DDDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

print_distribution(pms, env.get_develop(), 'Initialization', 3)
print_utilization(env.get_develop(), pms, 'Initialization')

dqn_loss = []
ppo_actor_loss = []
ppo_critic_loss = []

flag = False
n_reward = 0

return_list = []
start_time = time.time()
for i in range(1):
    if flag:
        break
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
        if flag:
            break
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0

            state = env.get_state()

            trajectory = {}
            trajectory_queue = []
            for _ in range(num_per_episodes):
                action_dqn = agent_dqn.take_action(state)
                action_ppo = agent_ppo.take_action(state)

                next_state, reward = env.step(action_dqn, action_ppo)
                done = False
                experience = (state, action_dqn, reward, next_state, done)

                n_step_transition = n_step_buffer.push(experience)
                if n_step_transition:
                    if RainbowConfig.use_prioritized_replay:
                        replay_buffer.add(n_step_transition)
                    else:
                        s, a, r, ns, d = n_step_transition
                        replay_buffer.add(s, a, r, ns)

                state = next_state
                episode_return += reward

                trajectory['states'] = state
                trajectory['actions'] = action_ppo
                trajectory['rewards'] = reward
                trajectory['next_states'] = next_state

                if reward == 0:
                    n_reward += 1
                if n_reward >= 50:
                    flag = True
                    break

                if replay_buffer.size() > minimal_size:
                    indices, batch, is_weights = sample_from_buffer(replay_buffer, batch_size,
                                                                    RainbowConfig.use_prioritized_replay)
                    b_s = [exp[0] for exp in batch]
                    b_a = [exp[1] for exp in batch]
                    b_r = [exp[2] for exp in batch]
                    b_ns = [exp[3] for exp in batch]

                    is_weights = torch.tensor(is_weights, dtype=torch.float).view(-1, 1).to(device)

                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'is_weights': is_weights,
                    }

                    dl, td_errors = agent_dqn.update(transition_dict)
                    al, cl = agent_ppo.update(trajectory)

                    dqn_loss.append(dl)
                    ppo_actor_loss.append(al)
                    ppo_critic_loss.append(cl)

                    replay_buffer.update_priorities(indices, td_errors)

            for t in n_step_buffer.flush():
                if RainbowConfig.use_prioritized_replay:
                    replay_buffer.add(t)
                else:
                    s, a, r, ns, d = t
                    replay_buffer.add(s, a, r, ns)

            return_list.append(episode_return)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
end_time = time.time()
run_time = end_time - start_time

hrl_develop = env.get_develop()
hrl_ei = Reward.ei(hrl_develop, nums_category=3)
hrl_si = Reward.si(hrl_develop, pms)
print("Time：", run_time)
print("Reward：", Reward.compute_reward(0, 0, hrl_ei, hrl_si))
print("HRL Reward：", 100.0 / (Reward.compute_reward(0, 0, hrl_ei, hrl_si) * run_time))

episodes_list = list(range(len(return_list)))
mv_return = moving_average(return_list)

print_returns(episodes_list, mv_return, "HRL", env_name)
print_distribution(pms, env.get_develop(), 'HRL', 3)
print_utilization(env.get_develop(), pms, 'HRL')

plt.figure(figsize=(12, 6))
plt.plot(moving_average(dqn_loss), label='DQN Loss')
plt.plot(moving_average(ppo_actor_loss), label='PPO Actor Loss')
plt.plot(moving_average(ppo_critic_loss), label='PPO Critic Loss')
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.title('Loss Curve for DQN and PPO')
plt.legend()
plt.show()

# actor_lr = 1e-3
# critic_lr = 1e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
#
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 100
# num_vms = 400
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
# replay_buffer = ReplayBuffer(buffer_size)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 50
# agent = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
# ppo_actor_loss = []
# ppo_critic_loss = []
#
# print_distribution(pms, env.get_develop(), 'Initialization', 3)
#
# return_list = []
# start_time = time.time()
# for i in range(1):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             state = env.get_state()
#
#             trajectory = {}
#
#             for _ in range(num_per_episodes):
#                 action = agent.take_action(state)
#                 next_state, reward = env.step_ppo(action)
#                 trajectory['states'] = state
#                 trajectory['actions'] = action
#                 trajectory['rewards'] = reward
#                 trajectory['next_states'] = next_state
#
#                 state = next_state
#                 episode_return += reward
#
#                 al, cl = agent.update(trajectory)
#
#             return_list.append(episode_return)
#             ppo_actor_loss.append(al)
#             ppo_critic_loss.append(cl)
#
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
# end_time = time.time()
# run_time = end_time - start_time
#
# ppo_develop = env.get_develop()
# ppo_ei = Reward.ei(ppo_develop, nums_category=3)
# ppo_si = Reward.si(ppo_develop, pms)
# print("Time：", run_time)
# print("Reward：", Reward.compute_reward(0, 0, ppo_ei, ppo_si))
# print("HRL Reward：", 100.0 / (Reward.compute_reward(0, 0, ppo_ei, ppo_si) * run_time))
#
# episodes_list = list(range(len(return_list)))
# mv_return = moving_average(return_list)
#
# print_returns(episodes_list, mv_return, "PPO", env_name)
# print_distribution(pms, env.get_develop(), 'PPO', 3)
#
# plt.figure(figsize=(12, 6))
# plt.plot(moving_average(ppo_actor_loss), label='PPO Actor Loss')
# plt.plot(moving_average(ppo_critic_loss), label='PPO Critic Loss')
# plt.xlabel('Training steps')
# plt.ylabel('Loss')
# plt.title('Loss Curve for DQN and PPO')
# plt.legend()
# plt.show()
#
#
# lr = 2e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
#
# epsilon = 0.5
# decay = 0.995
# min_epsilon = 0.01
#
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 100
# num_vms = 400
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
# replay_buffer = ReplayBuffer(buffer_size)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 100
# agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)

# print_distribution(pms, env.get_develop(), 'Initialization', 3)
#
# random_return_list = []
# return_list = []

# nopms = 0
# nom = 0
# ei = 0
# si = 0

# start_time = time.time()
# for i in range(1):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
            # random_return = 0
            # random_develop = env.get_develop()
            # state = env.get_state()

            # for _ in range(num_per_episodes):
            #     action = agent.take_action(state, epsilon)
            #     epsilon = max(min_epsilon, epsilon * decay)
            #     next_state, reward = env.step_DQN(action)
            #     replay_buffer.add(state, action, reward, next_state)
            #     state = next_state
            #     episode_return += reward
            #
            #     if replay_buffer.size() > minimal_size:
            #         b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
            #         transition_dict = {
            #             'states': b_s,
            #             'actions': b_a,
            #             'rewards': b_r,
            #             'next_states': b_ns
            #         }
            #         agent.update(transition_dict)

                # random_develop, random_reward, nopms, nom, ei, si = greedy_migration(pms,
                #                                                                  random_develop, nopms, nom, ei, si)
                # random_return += random_reward

            # return_list.append(episode_return)
            # random_return_list.append(random_return)

            # if (i_episode + 1) % 10 == 0:
            #     pbar.set_postfix({
            #         'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    # 'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
# end_time = time.time()
# run_time = end_time - start_time
#
# dqn_develop = env.get_develop()
# dqn_ei = Reward.ei(dqn_develop, nums_category=3)
# dqn_si = Reward.si(dqn_develop, pms)
# print("Time：", run_time)
# print("Reward：", Reward.compute_reward(0, 0, dqn_ei, dqn_si))
# print("DQN Reward：", 100.0 / (Reward.compute_reward(0, 0, dqn_ei, dqn_si) * run_time))

# random_ei = Reward.ei(random_develop, nums_category=3)
# random_si = Reward.si(random_develop, pms)
# print("Time：", run_time)
# print("Reward：", Reward.compute_reward(0, 0, random_ei, random_si))
# print("Greedy Reward：", 100.0 / (Reward.compute_reward(0, 0, random_ei, random_si) * run_time))

# episodes_list = list(range(len(return_list)))
# episodes_list = list(range(len(random_return_list)))
#
# mv_return = moving_average(return_list)
# mv_random_return = moving_average(random_return_list)
#
# print_returns(episodes_list, mv_return, "DQN", env_name)
# print_returns(episodes_list, mv_random_return, "Greedy", env_name)
#
# plt.figure()
# plt.plot(episodes_list, mv_return, label='DQN')
# plt.plot(episodes_list, mv_random_return, label='Greedy')
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN vs Greedy on {}'.format(env_name))
# plt.legend()
# plt.show()
#
# print_distribution(pms, env.get_develop(), 'DQN', 3)
# print_distribution(pms, random_develop, 'Greedy', 3)




# lr = 1e-3
# actor_lr = 1e-3
# critic_lr = 1e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
#
# epsilon = 0.5
# decay = 0.995
# min_epsilon = 0.01
#
# n_step = 3
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 100
# num_vms = 400
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_ppo = copy.deepcopy(env)
# env_dqn = copy.deepcopy(env)
#
# env_name = "VirtualMachineCluster"
#
# replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
# n_step_buffer = NStepBuffer(n_step, gamma=gamma)
# replay_buffer_dqn = ReplayBuffer(capacity=buffer_size)
#
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 5
#
# agent_dqn = RainbowDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
# agent_normal_dqn = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent_normal_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
#
# print_distribution(pms, env.get_develop(), 'Initialization', 3)
# print_utilization(env.get_develop(), pms, 'Initialization')
#
# dqn_loss = []
# ppo_actor_loss = []
# ppo_critic_loss = []
#
#
# return_list = []
# return_dqn_list = []
# return_ppo_list = []
# return_random_list = []
#
# nopms = 0
# nom = 0
# ei = 0
# si = 0
# for i in range(1):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return_hrl = 0
#             episode_return_dqn = 0
#             episode_return_ppo = 0
#             episode_return_random = 0
#
#             state = env.get_state()
#             state_dqn = env_dqn.get_state()
#             state_ppo = env_ppo.get_state()
#             random_develop = env.get_develop()
#
#             trajectory = {}
#             trajectory_queue = []
#
#             trajectory_ppo = {}
#             for _ in range(num_per_episodes):
#                 # HRL算法的训练
#                 action_dqn = agent_dqn.take_action(state)
#                 action_ppo = agent_ppo.take_action(state)
#
#                 next_state, reward = env.step(action_dqn, action_ppo)
#                 done = False
#                 experience = (state, action_dqn, reward, next_state, done)
#
#                 n_step_transition = n_step_buffer.push(experience)
#                 if n_step_transition:
#                     replay_buffer.add(n_step_transition)
#
#                 state = next_state
#                 episode_return_hrl += reward
#
#                 trajectory['states'] = state
#                 trajectory['actions'] = action_ppo
#                 trajectory['rewards'] = reward
#                 trajectory['next_states'] = next_state
#
#                 if replay_buffer.size() > minimal_size:
#                     indices, batch, is_weights = replay_buffer.sample(batch_size)
#                     b_s = [exp[0] for exp in batch]
#                     b_a = [exp[1] for exp in batch]
#                     b_r = [exp[2] for exp in batch]
#                     b_ns = [exp[3] for exp in batch]
#
#                     is_weights = torch.tensor(is_weights, dtype=torch.float).view(-1, 1).to(device)
#
#                     transition_dict = {
#                         'states': b_s,
#                         'actions': b_a,
#                         'rewards': b_r,
#                         'next_states': b_ns,
#                         'is_weights': is_weights,
#                     }
#
#                     dl, td_errors = agent_dqn.update(transition_dict)
#                     al, cl = agent_ppo.update(trajectory)
#
#                     dqn_loss.append(dl)
#                     ppo_actor_loss.append(al)
#                     ppo_critic_loss.append(cl)
#
#                     replay_buffer.update_priorities(indices, td_errors)
#
#                 #普通DQN算法的训练
#                 action = agent_normal_dqn.take_action(state_dqn, epsilon)
#                 epsilon = max(min_epsilon, epsilon * decay)
#                 next_state, reward = env_dqn.step_DQN(action)
#                 replay_buffer_dqn.add(state_dqn, action, reward, next_state)
#                 state_dqn = next_state
#                 episode_return_dqn += reward
#
#                 if replay_buffer_dqn.size() > minimal_size:
#                     b_s, b_a, b_r, b_ns = replay_buffer_dqn.sample(batch_size)
#                     transition_dict_dqn = {
#                         'states': b_s,
#                         'actions': b_a,
#                         'rewards': b_r,
#                         'next_states': b_ns
#                     }
#                     agent_normal_dqn.update(transition_dict_dqn)
#
#                 #贪心算法
#                 random_develop, random_reward, nopms, nom, ei, si = greedy_migration(pms, random_develop, nopms, nom,
#                                                                                      ei, si)
#                 episode_return_random += random_reward
#
#                 #PPO算法的训练
#                 action = agent_normal_ppo.take_action(state_ppo)
#                 next_state, reward = env_ppo.step_ppo(action)
#                 trajectory_ppo['states'] = state
#                 trajectory_ppo['actions'] = action
#                 trajectory_ppo['rewards'] = reward
#                 trajectory_ppo['next_states'] = next_state
#
#                 state_ppo = next_state
#                 episode_return_ppo += reward
#                 agent_normal_ppo.update(trajectory_ppo)
#
#             for t in n_step_buffer.flush():
#                 replay_buffer.add(t)
#
#             return_list.append(episode_return_hrl)
#             return_dqn_list.append(episode_return_dqn)
#             return_random_list.append(episode_return_random)
#             return_ppo_list.append(episode_return_ppo)
#
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
#
#
# episodes_list = list(range(len(return_list)))
# mv_return = moving_average(return_list)
#
# print_returns(episodes_list, mv_return, "HRL", env_name)
# print_distribution(pms, env.get_develop(), 'HRL', 3)
# print_utilization(env.get_develop(), pms, 'HRL')
#
# plt.figure(figsize=(12, 6))
# plt.plot(moving_average(dqn_loss), label='DQN Loss')
# plt.plot(moving_average(ppo_actor_loss), label='PPO Actor Loss')
# plt.plot(moving_average(ppo_critic_loss), label='PPO Critic Loss')
# plt.xlabel('Training steps')
# plt.ylabel('Loss')
# plt.title('Loss Curve for DQN and PPO')
# plt.legend()
# plt.show()
