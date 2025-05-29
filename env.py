import copy
import time
import torch
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reward import Reward
from utils import get_load, isdeploy, get_cpu_load, get_mem_load, sample_from_buffer
from DQN import ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer, RainbowDQN, DQN, DDDQN
from PPO import PPO
from config import RainbowConfig


def generate_pms(num_pms, pm_cpu_range, pm_mem_range):
    pms = []
    for pid in range(num_pms):
        cpu = np.random.randint(pm_cpu_range[0], pm_cpu_range[1])
        mem = np.random.randint(pm_mem_range[0], pm_mem_range[1])
        pms.append({'Pid': pid, 'Cpu': cpu, 'Mem': mem})
    return pms


def generate_vms(num_vms, pm_cpu_max=64, pm_mem_max=256):
    vms = []
    for vid in range(num_vms):
        cpu = int(abs(np.random.normal(loc=8, scale=4)))
        cpu = max(1, min(cpu, pm_cpu_max))

        mem = int(abs(np.random.normal(loc=16, scale=8)))
        mem = max(1, min(mem, pm_mem_max))

        vms.append({'Vid': vid, 'Cpu': cpu, 'Mem': mem})
    return vms


def cluster_vms(vms, n_cluster=3):
    X = np.array([[vm['Cpu'], vm['Mem']] for vm in vms])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_cluster, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    for i, vm in enumerate(vms):
        vm['Category'] = int(clusters[i])

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return vms, cluster_centers


class VirtualMachineClusterEnv:
    def __init__(self, pms, vms):
        self.nopms = 0
        self.nom = 0
        self.ei = 0
        self.si = 0
        self.num_pm = len(pms)
        self.pms = pms
        self.vms = vms
        self.develop = {pm['Pid']: [] for pm in self.pms}
        for vm in self.vms:
            while True:
                pm = np.random.choice(pms)
                if isdeploy(vm, pm, self.develop):
                    self.develop[pm['Pid']].append(vm)
                    break

    def get_develop(self):
        return self.develop

    def get_state(self, develop=None):
        state_matrix = np.zeros((len(self.pms), 3))
        if develop is None:
            develop = self.develop
        for pm_id, vm_list in develop.items():
            total_cpu = sum(vm['Cpu'] for vm in vm_list)
            total_mem = sum(vm['Mem'] for vm in vm_list)
            state_matrix[pm_id] = [total_cpu, total_mem, len(vm_list)]
        return np.array(state_matrix.flatten())

    def step(self, action_dqn, action_ppo):
        if action_dqn == 0:
            return self.get_state(self.develop), 0.0

        if action_dqn == 1:
            src_pm_id = int(action_ppo[0] * (self.num_pm - 1))

        if action_dqn == 2:
            sorted_pm = sorted(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']), reverse=True)
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        if action_dqn == 3:
            sorted_pm = sorted(self.pms, key=lambda pm: get_cpu_load(self.develop, pm['Pid']), reverse=True)
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        if action_dqn == 4:
            sorted_pm = sorted(self.pms, key=lambda pm: get_mem_load(self.develop, pm['Pid']), reverse=True)
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        dst_pm_id = int(action_ppo[2] * (self.num_pm - 1))
        length = len(self.develop[src_pm_id])

        if length == 0:
            return self.get_state(self.develop), -0.

        if dst_pm_id == src_pm_id:
            return self.get_state(self.develop), -0.2

        vm_id = int(action_ppo[1] * (length - 1))
        vm = self.develop[src_pm_id][vm_id]

        if not isdeploy(vm, self.pms[dst_pm_id], self.develop):
            return self.get_state(self.develop), -0.2

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(vm)
        next_develop[dst_pm_id].append(vm)
        next_nopms = Reward.nopms(next_develop)
        next_nom = Reward.nom(self.develop, src_pm_id, dst_pm_id)
        next_ei = Reward.ei(next_develop, nums_category=3)
        next_si = Reward.si(next_develop, self.pms)
        reward = Reward.compute_reward(next_nopms - self.nopms, next_nom - self.nom, next_ei - self.ei, next_si - self.si)

        if self.nopms == 0 and self.nom and self.ei == 0 and self.si == 0:
            self.develop = next_develop
            self.nopms = next_nopms
            self.nom = next_nom
            self.ei = next_ei
            self.si = next_si

            return self.get_state(next_develop), 0.0

        self.develop = next_develop
        self.nopms = next_nopms
        self.nom = next_nom
        self.ei = next_ei
        self.si = next_si

        cpu_load = get_cpu_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Cpu']
        mem_load = get_mem_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Mem']
        if cpu_load >= 0.8:
            reward -= 1
        if mem_load >= 0.8:
            reward -= 1

        return self.get_state(next_develop), reward

    def step_DQN(self, action_idx, device=None):
        max_load_pm = max(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']))['Pid']
        sorted_pm = sorted(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']))
        low_load_pm_list = [pm['Pid'] for pm in sorted_pm]

        vm = np.random.choice(self.develop[max_load_pm])
        vm_id = vm['Vid']
        dst_pm_index = action_idx
        src_pm_id = max_load_pm
        dst_pm_id = low_load_pm_list[dst_pm_index]
        target_vm = next((vm for vm in self.develop[src_pm_id] if vm['Vid'] == vm_id), None)

        next_develop = copy.deepcopy(self.develop)

        if src_pm_id == dst_pm_id:
            return self.get_state(self.develop), -0.2

        if not isdeploy(vm, sorted_pm[dst_pm_id], next_develop):
            return self.get_state(self.develop), -0.2

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(vm)
        next_develop[dst_pm_id].append(vm)
        next_nopms = Reward.nopms(next_develop)
        next_nom = Reward.nom(self.develop, src_pm_id, dst_pm_id)
        next_ei = Reward.ei(next_develop)
        next_si = Reward.si(next_develop, self.pms)
        reward = Reward.compute_reward(next_nopms - self.nopms, next_nom - self.nom, next_ei - self.ei,
                                       next_si - self.si)

        if self.nopms == 0 and self.nom and self.ei == 0 and self.si == 0:
            self.develop = next_develop
            self.nopms = next_nopms
            self.nom = next_nom
            self.ei = next_ei
            self.si = next_si

            return self.get_state(next_develop), 0.0

        self.develop = next_develop
        self.nopms = next_nopms
        self.nom = next_nom
        self.ei = next_ei
        self.si = next_si

        cpu_load = get_cpu_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Cpu']
        mem_load = get_mem_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Mem']
        if cpu_load >= 0.8:
            reward -= 1
        if mem_load >= 0.8:
            reward -= 1

        return self.get_state(next_develop), reward

    def step_ppo(self, action, device=None):
        src_pm_id = int(action[0] * (self.num_pm - 1))
        dst_pm_id = int(action[2] * (self.num_pm - 1))
        length = len(self.develop[src_pm_id])

        if dst_pm_id == src_pm_id or length == 0:
            return self.get_state(self.develop), -0.2

        vm_id = int(action[1] * (length - 1))
        vm = self.develop[src_pm_id][vm_id]

        if not isdeploy(vm, self.pms[dst_pm_id], self.develop):
            return self.get_state(self.develop), -0.2

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(vm)
        next_develop[dst_pm_id].append(vm)
        next_nopms = Reward.nopms(next_develop)
        next_nom = Reward.nom(self.develop, src_pm_id, dst_pm_id)
        next_ei = Reward.ei(next_develop)
        next_si = Reward.si(next_develop, self.pms)
        reward = Reward.compute_reward(next_nopms - self.nopms, next_nom - self.nom, next_ei - self.ei,
                                       next_si - self.si)

        if self.nopms == 0 and self.nom and self.ei == 0 and self.si == 0:
            self.develop = next_develop
            self.nopms = next_nopms
            self.nom = next_nom
            self.ei = next_ei
            self.si = next_si

            return self.get_state(next_develop), 0.0

        self.develop = next_develop
        self.nopms = next_nopms
        self.nom = next_nom
        self.ei = next_ei
        self.si = next_si

        cpu_load = get_cpu_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Cpu']
        mem_load = get_mem_load(self.develop, dst_pm_id) / self.pms[dst_pm_id]['Mem']
        if cpu_load >= 0.8:
            reward -= 1
        if mem_load >= 0.8:
            reward -= 1

        return self.get_state(next_develop), reward


def train_all_agents(buffer_size, n_step, gamma, state_dim, action_dim, lr, epsilon, target_update, device, actor_lr,
                     critic_lr, lmbda, epochs, eps, num_episodes, num_per_episodes, min_epsilon, decay, env,
                     minimal_size, batch_size):
    # 初始化三个 replay buffer 和 N-step buffer
    replay_buffer_dqn = ReplayBuffer(buffer_size)
    replay_buffer_rainbow = PrioritizedReplayBuffer(capacity=buffer_size)
    n_step_buffer = NStepBuffer(n_step, gamma=gamma)

    # 初始化三个 agent
    agent_dqn = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    agent_rainbow = RainbowDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent_ppo_rainbow = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list_dqn = []
    return_list_ppo = []
    return_list_rainbow_ppo = []

    for i in range(1):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i + 1}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # ---- DQN ----
                state = env.get_state()
                episode_return_dqn = 0
                for _ in range(num_per_episodes):
                    action = agent_dqn.take_action(state, epsilon)
                    epsilon = max(min_epsilon, epsilon * decay)
                    next_state, reward = env.step_DQN(action)
                    replay_buffer_dqn.add(state, action, reward, next_state)
                    state = next_state
                    episode_return_dqn += reward

                    if replay_buffer_dqn.size() > minimal_size:
                        b_s, b_a, b_r, b_ns = replay_buffer_dqn.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns}
                        agent_dqn.update(transition_dict)
                return_list_dqn.append(episode_return_dqn)

                # ---- PPO ----
                state = env.get_state()
                episode_return_ppo = 0
                for _ in range(num_per_episodes):
                    action = agent_ppo.take_action(state)
                    next_state, reward = env.step_ppo(action)
                    trajectory = {
                        'states': state,
                        'actions': action,
                        'rewards': reward,
                        'next_states': next_state
                    }
                    al, cl = agent_ppo.update(trajectory)
                    state = next_state
                    episode_return_ppo += reward
                return_list_ppo.append(episode_return_ppo)

                # ---- Rainbow DQN + PPO ----
                state = env.get_state()
                episode_return_rainbow = 0
                for _ in range(num_per_episodes):
                    action_dqn = agent_rainbow.take_action(state)
                    action_ppo = agent_ppo_rainbow.take_action(state)

                    next_state, reward = env.step(action_dqn, action_ppo)
                    experience = (state, action_dqn, reward, next_state, False)

                    n_step_transition = n_step_buffer.push(experience)
                    if n_step_transition:
                        replay_buffer_rainbow.add(n_step_transition)

                    trajectory = {
                        'states': state,
                        'actions': action_ppo,
                        'rewards': reward,
                        'next_states': next_state
                    }

                    state = next_state
                    episode_return_rainbow += reward

                    if replay_buffer_rainbow.size() > minimal_size:
                        indices, batch, is_weights = replay_buffer_rainbow.sample(batch_size)
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
                            'is_weights': is_weights
                        }
                        dl, td_errors = agent_rainbow.update(transition_dict)
                        al, cl = agent_ppo_rainbow.update(trajectory)
                        replay_buffer_rainbow.update_priorities(indices, td_errors)

                for t in n_step_buffer.flush():
                    replay_buffer_rainbow.add(t)

                return_list_rainbow_ppo.append(episode_return_rainbow)

                # ---- 日志打印 ----
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'DQN Return': '%.2f' % np.mean(return_list_dqn[-10:]),
                        'PPO Return': '%.2f' % np.mean(return_list_ppo[-10:]),
                        'Rainbow+PPO': '%.2f' % np.mean(return_list_rainbow_ppo[-10:])
                    })
                pbar.update(1)

    return return_list_dqn, return_list_ppo, return_list_rainbow_ppo


def train(gamma, lr, actor_lr, critic_lr, epsilon, lmbda, epochs, eps, num_episodes, num_per_episodes):
    if not RainbowConfig.use_multi_step and not RainbowConfig.no_rainbow:
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

    if RainbowConfig.use_prioritized_replay and not RainbowConfig.no_rainbow:
        replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
    else:
        replay_buffer = ReplayBuffer(capacity=buffer_size)
    n_step_buffer = NStepBuffer(n_step, gamma=gamma)

    state_dim = len(env.get_state(env.develop))
    action_dim = 5

    if RainbowConfig.use_distributional_rl and not RainbowConfig.no_rainbow:
        agent_dqn = RainbowDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    elif RainbowConfig.no_rainbow:
        agent_dqn = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    else:
        agent_dqn = DDDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    flag = False
    n_reward = 0

    start_time = time.time()
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
                for _ in range(num_per_episodes):
                    action_dqn = agent_dqn.take_action(state, epsilon)
                    if not RainbowConfig.use_noisy_net or RainbowConfig.no_rainbow:
                        epsilon = max(0.01, epsilon * 0.995)
                    action_ppo = agent_ppo.take_action(state)

                    next_state, reward = env.step(action_dqn, action_ppo)
                    done = False
                    experience = (state, action_dqn, reward, next_state, done)

                    n_step_transition = n_step_buffer.push(experience)
                    if n_step_transition:
                        if RainbowConfig.use_prioritized_replay and not RainbowConfig.no_rainbow:
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
                        if not RainbowConfig.no_rainbow:
                            dl, td_errors = agent_dqn.update(transition_dict)
                            replay_buffer.update_priorities(indices, td_errors)
                        else:
                            agent_dqn.update(transition_dict)
                        al, cl = agent_ppo.update(trajectory)


                for t in n_step_buffer.flush():
                    if RainbowConfig.use_prioritized_replay and not RainbowConfig.no_rainbow:
                        replay_buffer.add(t)
                    else:
                        s, a, r, ns, d = t
                        replay_buffer.add(s, a, r, ns)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1)})
                pbar.update(1)
    end_time = time.time()
    run_time = end_time - start_time

    hrl_develop = env.get_develop()
    hrl_ei = Reward.ei(hrl_develop, nums_category=3)
    hrl_si = Reward.si(hrl_develop, pms)

    return (run_time, Reward.compute_reward(0, 0, hrl_ei, hrl_si),
            100.0 / (Reward.compute_reward(0, 0, hrl_ei, hrl_si) * run_time))
