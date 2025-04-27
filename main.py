import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from env import generate_pms, generate_vms, cluster_vms, VirtualMachineClusterEnv
from DQN import DQN, ReplayBuffer, DoubleDQN, DDDQN
from PPO import PPO
from utils import print_distribution, moving_average, print_returns, print_utilization
from contrast import greedy_migration

# 设定超参数
lr = 2e-3
actor_lr = 1e-4
critic_lr = 1e-4
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

target_update = 10
buffer_size = 10000
minimal_size = 100
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_pms = 50
num_vms = 200
pm_cpu_range = (32, 128)
pm_mem_range = (128, 512)

pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
vms = generate_vms(num_vms)
vms, centers = cluster_vms(vms, 3)

env = VirtualMachineClusterEnv(pms, vms)
env_name = "VirtualMachineCluster"
replay_buffer = ReplayBuffer(buffer_size)

state_dim = len(env.get_state(env.develop))
action_dim = 5

agent_dqn = DDDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

print_distribution(pms, env.get_develop(), 'Initialization')
print_utilization(env.get_develop(), pms, 'Initialization')

dqn_loss = []
ppo_actor_loss = []
ppo_critic_loss = []

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.get_state()

            trajectory = {}
            for _ in range(num_per_episodes):
                action_dqn = agent_dqn.take_action(state, epsilon)
                action_ppo = agent_ppo.take_action(state)

                epsilon = max(min_epsilon, epsilon * decay)
                next_state, reward = env.step(action_dqn, action_ppo)

                replay_buffer.add(state, action_dqn, reward, next_state)

                state = next_state
                episode_return += reward

                trajectory['states'] = state
                trajectory['actions'] = action_ppo
                trajectory['rewards'] = reward
                trajectory['next_states'] = next_state

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns
                    }
                    dl = agent_dqn.update(transition_dict)
                    al, cl = agent_ppo.update(trajectory)

                    dqn_loss.append(dl)
                    ppo_actor_loss.append(al)
                    ppo_critic_loss.append(cl)

            return_list.append(episode_return)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))
mv_return = moving_average(return_list)

print_returns(episodes_list, mv_return, "HRL", env_name)
print_distribution(pms, env.get_develop(), 'HRL')
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
# num_pms = 50
# num_vms = 200
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
# replay_buffer = ReplayBuffer(buffer_size, device)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 50
# agent = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
#
# print_distribution(pms, env.get_develop(), 'Initialization')
#
# return_list = []
# for i in range(10):
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
#                 print(reward)
#                 trajectory['states'] = state
#                 trajectory['actions'] = action
#                 trajectory['rewards'] = reward
#                 trajectory['next_states'] = next_state
#
#                 state = next_state
#                 episode_return += reward
#
#                 agent.update(trajectory)
#
#             return_list.append(episode_return)
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
# print_returns(episodes_list, mv_return, "PPO", env_name)
# print_distribution(pms, env.get_develop(), 'PPO')
