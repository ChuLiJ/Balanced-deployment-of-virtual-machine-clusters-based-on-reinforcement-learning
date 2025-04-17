import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reward import Reward
from utils import get_load


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
        self.pms = pms
        self.vms = vms
        self.develop = {pm['Pid']: [] for pm in self.pms}
        for vm in self.vms:
            while True:
                pm = np.random.choice(pms)
                used_cpu = sum(v['Cpu'] for v in self.develop[pm['Pid']])
                used_mem = sum(v['Mem'] for v in self.develop[pm['Pid']])
                if (used_cpu + vm['Cpu'] <= pm['Cpu']) and (used_mem + vm['Mem'] <= pm['Mem']):
                    self.develop[pm['Pid']].append(vm)
                    break

    def get_state(self, develop=None):
        state_matrix = np.zeros((len(self.pms), 3))
        if develop is None:
            develop = self.develop
        for pm_id, vm_list in develop.items():
            total_cpu = sum(vm['Cpu'] for vm in vm_list)
            total_mem = sum(vm['Mem'] for vm in vm_list)
            state_matrix[pm_id] = [total_cpu, total_mem, len(vm_list)]
        return np.array(state_matrix.flatten())

    def step(self, action_idx):
        max_load_pm = max(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']))['Pid']
        sorted_pm = sorted(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']))
        low_load_pm_list = [pm['Pid'] for pm in sorted_pm[:5]]

        index = int(np.random.rand() * len(self.develop[max_load_pm]))
        vm_id = self.develop[max_load_pm][index]['Vid']
        dst_pm_index = action_idx % 5
        src_pm_id = max_load_pm
        dst_pm_id = low_load_pm_list[dst_pm_index]

        target_vm = next((vm for vm in self.develop[src_pm_id] if vm['Vid'] == vm_id), None)
        if target_vm is None:
            raise ValueError("Target Vm is None")

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(target_vm)
        next_develop[dst_pm_id].append(target_vm)
        reward = Reward.compute_reward(self.develop, next_develop, self.pms,
                                       (0, 50), (0, 200), (-1.0, 0.0), (0.0, 2.0))
        self.develop = next_develop
        return self.get_state(next_develop), reward
