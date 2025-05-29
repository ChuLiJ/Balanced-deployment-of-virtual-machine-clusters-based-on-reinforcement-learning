import torch
import numpy as np
import matplotlib.pyplot as plt

from django.db.models import Prefetch

from ..models import Deploy, PhysicalMachine, VirtualMachine


def iscandeploy(vm, pm_obj):
    deploy_list = Deploy.objects.filter(pm=pm_obj)
    if not deploy_list.exists():
        return True

    used_cpu = 0
    used_mem = 0
    for deploy in deploy_list:
        used_cpu += deploy.vm.cpu
        used_mem += deploy.vm.memory
    if (used_cpu + vm['Cpu'] <= pm_obj.cpu) and (used_mem + vm['Mem'] <= pm_obj.memory):
        return True
    return False


def isdeploy(vm_obj, pm_obj):
    deploy_list = Deploy.objects.filter(pm=pm_obj)
    if not deploy_list.exists():
        return True

    used_cpu = 0
    used_mem = 0
    for deploy in deploy_list:
        used_cpu += deploy.vm.cpu
        used_mem += deploy.vm.memory
    if (used_cpu + vm_obj.cpu <= pm_obj.cpu) and (used_mem + vm_obj.memory <= pm_obj.memory):
        return True
    return False


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)


def get_load(pm):
    return (pm.used_cpu + pm.used_mem) / 2


def print_distribution(name="current state"):
    pms = PhysicalMachine.objects.prefetch_related(
        Prefetch('deployments', queryset=Deploy.objects.select_related('vm'))
    )

    plt.figure()
    category_counts = {pm.pid.hex: {i: 0 for i in range(3)} for pm in pms}

    for pm in pms:
        for deploy in pm.deployments.all():
            vm = deploy.vm
            if vm and 0 <= vm.category < 3:
                category_counts[pm.pid.hex][vm.category] += 1

    pm_ids = list(category_counts.keys())
    bottom = np.zeros(len(pm_ids))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / 3) for i in range(3)]

    for category in range(3):
        counts = [category_counts[pm_id][category] for pm_id in pm_ids]
        plt.bar(pm_ids, counts, bottom=bottom, color=colors[category], label=f'Category {category}')
        bottom += np.array(counts)

    plt.xlabel('Physical Machine ID')
    plt.ylabel('Number of Virtual Machines')
    plt.title(f'Number of VMs on Each PM by Category based on {name}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def moving_average(a, window_size=9):
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


def print_utilization(name="current state"):
    pms = list(PhysicalMachine.objects.prefetch_related(
        Prefetch('deployments', queryset=Deploy.objects.select_related('vm'))
    ))

    cpu_utilization = []
    mem_utilization = []
    labels = []

    for pm in pms:
        used_cpu = sum(d.vm.cpu for d in pm.deployments.all() if d.vm)
        used_mem = sum(d.vm.memory for d in pm.deployments.all() if d.vm)
        total_cpu = pm.cpu
        total_mem = pm.memory
        labels.append(pm.name)

        cpu_utilization.append(used_cpu / total_cpu * 100 if total_cpu else 0)
        mem_utilization.append(used_mem / total_mem * 100 if total_mem else 0)

    x = np.arange(len(pms))
    bar_width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(x - bar_width / 2, cpu_utilization, bar_width, label='CPU Utilization', color='b')
    ax.bar(x + bar_width / 2, mem_utilization, bar_width, label='Memory Utilization', color='r')

    ax.set_xlabel('Physical Machine')
    ax.set_ylabel('Utilization Percentage (%)')
    ax.set_title(f'CPU and Memory Utilization based on {name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

