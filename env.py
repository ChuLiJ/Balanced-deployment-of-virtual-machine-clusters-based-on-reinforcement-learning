import copy
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reward import Reward
from utils import get_load, isdeploy, get_cpu_load, get_mem_load


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
            sorted_pm = sorted(self.pms, key=lambda pm: get_load(self.develop, pm['Pid']))
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        if action_dqn == 3:
            sorted_pm = sorted(self.pms, key=lambda pm: get_cpu_load(self.develop, pm['Pid']))
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        if action_dqn == 4:
            sorted_pm = sorted(self.pms, key=lambda pm: get_mem_load(self.develop, pm['Pid']))
            src_pm_id = sorted_pm[int(action_ppo[0] * 5)]['Pid']

        dst_pm_id = int(action_ppo[2] * (self.num_pm - 1))
        length = len(self.develop[src_pm_id])

        if length == 0:
            return self.get_state(self.develop), -0.1

        if dst_pm_id == src_pm_id:
            return self.get_state(self.develop), -0.5

        vm_id = int(action_ppo[1] * (length - 1))
        vm = self.develop[src_pm_id][vm_id]

        if not isdeploy(vm, self.pms[dst_pm_id], self.develop):
            return self.get_state(self.develop), -0.1

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(vm)
        next_develop[dst_pm_id].append(vm)
        next_nopms = Reward.nopms(next_develop)
        next_ei = Reward.ei(next_develop)
        next_si = Reward.si(next_develop, self.pms)
        reward = Reward.compute_reward(next_nopms - self.nopms, 0, next_ei - self.ei, next_si - self.si)

        if self.nopms == 0 and self.ei == 0 and self.si == 0:
            self.develop = next_develop
            self.nopms = next_nopms
            self.ei = next_ei
            self.si = next_si

            return self.get_state(next_develop), 0.0

        self.develop = next_develop
        self.nopms = next_nopms
        self.ei = next_ei
        self.si = next_si

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
            return self.get_state(self.develop), -1.0

        if not isdeploy(vm, sorted_pm[dst_pm_id], next_develop):
            return self.get_state(self.develop), -0.5

        next_develop[src_pm_id].remove(target_vm)
        next_develop[dst_pm_id].append(target_vm)
        reward = Reward.compute_reward(self.develop, next_develop, self.pms,
                                       (0, 50), (0, 200), (-1.0, 0.0), (0.0, 2.0))
        self.develop = next_develop
        return self.get_state(next_develop), reward

    def step_ppo(self, action, device=None):
        src_pm_id = int(action[0] * (self.num_pm - 1))
        dst_pm_id = int(action[2] * (self.num_pm - 1))
        length = len(self.develop[src_pm_id])

        if dst_pm_id == src_pm_id or length == 0:
            return self.get_state(self.develop), -0.5

        vm_id = int(action[1] * (length - 1))
        vm = self.develop[src_pm_id][vm_id]

        if not isdeploy(vm, self.pms[dst_pm_id], self.develop):
            return self.get_state(self.develop), -0.0

        next_develop = copy.deepcopy(self.develop)
        next_develop[src_pm_id].remove(vm)
        next_develop[dst_pm_id].append(vm)
        next_nopms = Reward.nopms(next_develop)
        next_ei = Reward.ei(next_develop)
        next_si = Reward.si(next_develop, self.pms)
        reward = Reward.compute_reward(next_nopms-self.nopms, 0, next_ei-self.ei, next_si-self.si)

        self.develop = next_develop
        self.nopms = next_nopms
        self.ei = next_ei
        self.si = next_si

        return self.get_state(next_develop), reward
