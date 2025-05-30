import torch
import numpy as np
import matplotlib.pyplot as plt

from config import RainbowConfig


def sample_from_buffer(buffer, batch_size, use_per=True):
    if use_per and not RainbowConfig.no_rainbow:
        # 如果启用 PER（Prioritized Replay）
        indices, experiences, is_weights = buffer.sample(batch_size)
    else:
        # 否则用普通 ReplayBuffer 包装成类似格式
        states, actions, rewards, next_states = buffer.sample(batch_size)
        experiences = list(zip(states, actions, rewards, next_states))
        indices = None
        is_weights = np.ones((batch_size, 1))  # 等权重
    return indices, experiences, is_weights

def isdeploy(vm, pm, develop):
    used_cpu = sum(v['Cpu'] for v in develop[pm['Pid']])
    used_mem = sum(v['Mem'] for v in develop[pm['Pid']])
    if (used_cpu + vm['Cpu'] <= pm['Cpu']) and (used_mem + vm['Mem'] <= pm['Mem']):
        return True
    return False


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)


def get_load(develop, pm_id):
    return (sum(vm['Cpu'] for vm in develop[pm_id]) + sum(vm['Mem'] for vm in develop[pm_id])) / 2


def get_cpu_load(develop, pm_id):
    return sum(vm['Cpu'] for vm in develop[pm_id])


def get_mem_load(develop, pm_id):
    return sum(vm['Mem'] for vm in develop[pm_id])


def print_distribution(pms, develop, name, num_category):
    plt.figure()
    category_counts = {pm['Pid']: {i: 0 for i in range(num_category)} for pm in pms}
    for pm_id, vm_list in develop.items():
        for vm in vm_list:
            category_counts[pm_id][vm['Category']] += 1

    pm_ids = list(category_counts.keys())
    bottom = np.zeros(len(pm_ids))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / num_category) for i in range(num_category)]

    for category in range(num_category):
        counts = [category_counts[pm_id][category] for pm_id in pm_ids]
        plt.bar(pm_ids, counts, bottom=bottom, color=colors[category], label=f'Category {category}')
        bottom += np.array(counts)

    plt.xlabel('Physical Machine ID')
    plt.ylabel('Number of Virtual Machines')
    plt.title('Number of VMs on Each PM by Category based on {}'.format(name))
    plt.legend()
    plt.show()


def moving_average(a, window_size=9):
    if isinstance(a[0], torch.Tensor):
        a = [x.detach().cpu().item() for x in a]
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def print_returns(episodes_list, return_list, name, env_name):
    plt.figure()
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('{0} on {1}'.format(name, env_name))
    plt.show()


def print_utilization(develop, pms, name):
    cpu_utilization = []
    mem_utilization = []
    for pm in pms:
        cpu_utilization.append(get_cpu_load(develop, pm['Pid']) / pm['Cpu'] * 100)
        mem_utilization.append(get_mem_load(develop, pm['Pid']) / pm['Mem'] * 100)

    x = range(len(pms))
    bar_width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar([i - bar_width / 2 for i in x], cpu_utilization, bar_width, label='CPU Utilization', color='b')
    ax.bar([i + bar_width / 2 for i in x], mem_utilization, bar_width, label='Memory Utilization', color='r')

    ax.set_xlabel('Physical Machine PID')
    ax.set_ylabel('Utilization Percentage (%)')
    ax.set_title('CPU and Memory Utilization based on {}'.format(name))
    ax.legend()

    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def __len__(self):
        return self.write if self.write <= self.capacity else self.capacity


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_feature, out_feature, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_feature
        self.out_features = out_feature

        self.weight_mu = torch.nn.Parameter(torch.empty(out_feature, in_feature))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_feature, in_feature))
        self.register_buffer('weight_epsilon', torch.empty(out_feature, in_feature))

        self.bias_mu = torch.nn.Parameter(torch.empty(out_feature))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_feature))
        self.register_buffer('bias_epsilon', torch.empty(out_feature))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())




# pms = generate_pms(50, (8, 64), (32, 256))
# vms = generate_vms(200, 64, 256)
# vms, centers = cluster_vms(vms, 3)

# env = VirtualMachineClusterEnv(pms, vms)
#
# pm_cpus = [pm['Cpu'] for pm in pms]
# pm_mems = [pm['Mem'] for pm in pms]
#
# plt.scatter(pm_cpus, pm_mems, alpha=0.6)
# plt.xlabel("CPU Cores")
# plt.ylabel("Memory (GB)")
# plt.title("Randomly Generated PMs")
# plt.show()
#
# cpus = [vm['Cpu'] for vm in vms]
# mems = [vm['Mem'] for vm in vms]
#
# plt.scatter(cpus, mems, alpha=0.6)
# plt.xlabel("CPU Cores")
# plt.ylabel("Memory (GB)")
# plt.title("Randomly Generated VMs (Before Clustering)")
# plt.show()
#
# categories = [vm["Category"] for vm in vms]
# plt.scatter(cpus, mems, c=categories, cmap='viridis', alpha=0.6)
# plt.xlabel("CPU Cores")
# plt.ylabel("Memory (GB)")
# plt.title("VMs After K-means Clustering (3 Categories)")
# plt.colorbar(label="Category")
# plt.show()
