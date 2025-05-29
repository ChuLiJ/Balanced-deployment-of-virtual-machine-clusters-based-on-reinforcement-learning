import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from .env import generate_pms, generate_vms, cluster_vms, VirtualMachineClusterEnv
from .agent import PPO, RainbowDQN, NStepBuffer, PrioritizedReplayBuffer
from .utils import print_distribution, moving_average, print_returns, print_utilization
from .contrast import greedy_migration
from ..models import RLHyperParameter, PhysicalMachine, VirtualMachine


def train_model(user):
    hyperparameter_obj = RLHyperParameter.objects.filter(user__in=[user.name, 'admin'])

    num_episodes = 1000
    num_per_episodes = 100
    epsilon = 0.01

    buffer_size = 10000
    minimal_size = 100
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pms = PhysicalMachine.objects.all()
    vms = VirtualMachine.objects.all()

    env = VirtualMachineClusterEnv(pms, vms)
    env_name = "VirtualMachineCluster"

    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
    n_step_buffer = NStepBuffer(hyperparameter_obj.n_step, gamma=hyperparameter_obj.gamma)

    state_dim = len(env.get_state())
    action_dim = 5

    agent_dqn = RainbowDQN(state_dim, action_dim, hyperparameter_obj.lr, hyperparameter_obj.gamma, epsilon,
                           hyperparameter_obj.target_update, device)
    agent_ppo = PPO(state_dim, hyperparameter_obj.actor_lr, hyperparameter_obj.critic_lr, hyperparameter_obj.lmbda,
                    hyperparameter_obj.epochs, hyperparameter_obj.eps, hyperparameter_obj.gamma, device)

    flag = False
    n_reward = 0

    return_list = []
    for i in range(1):
        if flag:
            break
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i + 1}') as pbar:
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
                        replay_buffer.add(n_step_transition)

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
                        indices, batch, is_weights = replay_buffer.sample(batch_size)
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
                        agent_ppo.update(trajectory)

                        replay_buffer.update_priorities(indices, td_errors)

                for t in n_step_buffer.flush():
                    replay_buffer.add(t)

                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

