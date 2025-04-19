import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from env import generate_pms, generate_vms, cluster_vms, VirtualMachineClusterEnv
from DQN import DQN, ReplayBuffer
from utils import print_distribution, moving_average, print_returns
from contrast import greedy_migration

# 设定超参数
lr = 2e-3
num_episodes = 1000
num_per_episodes = 100
hidden_dim = 128
gamma = 0.98

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
action_dim = 50
agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)

print_distribution(pms, env.get_develop(), 'Initialization')

random_return_list = []
return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            random_return = 0
            random_develop = env.get_develop()
            state = env.get_state()

            for _ in range(num_per_episodes):
                action = agent.take_action(state, epsilon)
                epsilon = max(min_epsilon, epsilon * decay)
                next_state, reward = env.step(action)
                replay_buffer.add(state, action, reward, next_state)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns
                    }
                    agent.update(transition_dict)

                random_develop, random_reward = greedy_migration(pms, random_develop)
                random_return += random_reward

            return_list.append(episode_return)
            random_return_list.append(random_return)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))

mv_return = moving_average(return_list)
mv_random_return = moving_average(random_return_list)

print_returns(episodes_list, mv_return, "DQN", env_name)
print_returns(episodes_list, mv_random_return, "Greedy", env_name)

plt.figure()
plt.plot(episodes_list, mv_return, label='DQN')
plt.plot(episodes_list, mv_random_return, label='Greedy')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN vs Greedy on {}'.format(env_name))
plt.legend()
plt.show()

print_distribution(pms, env.get_develop(), 'DQN')
print_distribution(pms, random_develop, 'Greedy')
