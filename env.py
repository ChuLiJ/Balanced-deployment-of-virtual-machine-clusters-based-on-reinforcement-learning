import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reward import Reward


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
        self.state = {pm['Pid']: [] for pm in self.pms}
        for vm in self.vms:
            while True:
                pm = np.random.choice(pms)
                used_cpu = sum(v['Cpu'] for v in self.state[pm['Pid']])
                used_mem = sum(v['Mem'] for v in self.state[pm['Pid']])
                if (used_cpu + vm['Cpu'] <= pm['Cpu']) and (used_mem + vm['Mem'] <= pm['Mem']):
                    self.state[pm['Pid']].append(vm)
                    break

    def step(self, action_idx):
        action_dim = len(pms) * (len(pms) - 1) * len(vms) - 1
        if action_idx > action_dim:
            raise ValueError("Invalid action index")
        vm_id = action_idx // len(pms) * (len(pms) - 1)
        remainder = action_idx % (len(pms) * (len(pms) - 1))
        src_pm_id = remainder // (len(pms) - 1)
        dst_pm_id = remainder % (len(pms) - 1)
        target_vm = None
        for vm in self.state[src_pm_id]:
            if vm['Vid'] == vm_id:
                target_vm = vm
                break
        if target_vm is None:
            raise ValueError("Target Vm is None")
        next_state = copy.deepcopy(self.state)
        next_state[src_pm_id].remove(target_vm)
        next_state[dst_pm_id].append(target_vm)
        reward = Reward.compute_reward(self.state, next_state, self.pms)
        done = False


pms = generate_pms(50, (8, 64), (32, 256))
vms = generate_vms(200, 64, 256)
vms, centers = cluster_vms(vms, 3)

env = VirtualMachineClusterEnv(pms, vms)


# cpus = [vm['Cpu'] for vm in vms]
# mems = [vm['Mem'] for vm in vms]
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


# def generate_requests(time_steps, vms, request_rate=0.3, avg_lifetime=5):
#     requests = []
#     active_vms = {}
#
#     for t in range(time_steps):
#         # 新请求虚拟机
#         if np.random.rand() < request_rate:
#             vm = np.random.choice(vms)
#             lifetime = np.random.exponential(avg_lifetime)
#             requests.append({
#                 "time": t,
#                 "vm": vm,
#                 "type": "start",
#                 "lifetime": int(lifetime)
#             })
#             active_vms[vm["Vid"]] = t + lifetime
#         # 移除过期虚拟机
#         expired_vids = [vid for vid, end_time in active_vms.items() if end_time <= t]
#         for vid in expired_vids:
#             requests.append({
#                 "time": t,
#                 "vm": next(vm for vm in vms if vm["Vid"] == vid),
#                 "type": "stop"
#             })
#             del active_vms[vid]
#     return requests
