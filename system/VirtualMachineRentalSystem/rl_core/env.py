import copy
import random
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .reward import Reward
from .utils import get_load, isdeploy, iscandeploy
from ..models import PhysicalMachine, VirtualMachine, Deploy


scaler = None
kmeans = None


def generate_pms(num_pms, pm_cpu_range, pm_mem_range):
    pm_objs = []
    count = PhysicalMachine.objects.count()
    for pid in range(num_pms):
        cpu = np.random.randint(pm_cpu_range[0], pm_cpu_range[1])
        mem = np.random.randint(pm_mem_range[0], pm_mem_range[1])
        pm_obj = PhysicalMachine.objects.create(
            cpu=cpu,
            memory=mem,
            name=f"PM-{pid+count+1}",
        )
        pm_objs.append(pm_obj)
    return pm_objs


def generate_vms(num_vms, pm_objs, pm_cpu_max=64, pm_mem_max=256, user=None):
    vms = []
    count = VirtualMachine.objects.count()
    for vid in range(num_vms):
        cpu = int(abs(np.random.normal(loc=8, scale=4)))
        cpu = max(1, min(cpu, pm_cpu_max))

        mem = int(abs(np.random.normal(loc=16, scale=8)))
        mem = max(1, min(mem, pm_mem_max))

        vms.append({'Vid': vid+count, 'Cpu': cpu, 'Mem': mem})

    vms = cluster_vms(vms, 3)
    for vm in vms:
        pm_obj = random.choice(pm_objs)
        while not iscandeploy(vm, pm_obj):
            pm_obj = random.choice(pm_objs)
        if user:
            vm_obj = VirtualMachine.objects.create(
                cpu=vm['Cpu'],
                memory=vm['Mem'],
                name=f"VM-{vm['Vid']+1}",
                category=vm['Category'],
                deploy_on=pm_obj,
                user=user,
            )

            deploy_obj = Deploy.objects.create(
                pm=pm_obj,
                vm=vm_obj,
                message="创建成功",
                method="Random",
                user=user,
            )

        else:
            vm_obj = VirtualMachine.objects.create(
                cpu=vm['Cpu'],
                memory=vm['Mem'],
                name=f"VM-{vm['Vid'] + 1}",
                category=vm['Category'],
                deploy_on=pm_obj,
            )

            deploy_obj = Deploy.objects.create(
                pm=pm_obj,
                vm=vm_obj,
                message="创建成功",
                method="Random",
            )


def cluster_vms(vms, n_cluster=3):
    global scaler, kmeans
    X = np.array([[vm['Cpu'], vm['Mem']] for vm in vms])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_cluster, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    for i, vm in enumerate(vms):
        vm['Category'] = int(clusters[i])

    return vms


def classify_new_vm(cpu, mem):
    global scaler, kmeans
    if scaler is None or kmeans is None:
        raise ValueError("请先调用 cluster_vms 训练模型")

    X_new = np.array([[cpu, mem]])
    X_new_scaled = scaler.transform(X_new)
    return int(kmeans.predict(X_new_scaled)[0])


class VirtualMachineClusterEnv:
    def __init__(self, pms, vms):
        self.nopms = 0
        self.nom = 0
        self.ei = 0
        self.si = 0
        self.num_pm = PhysicalMachine.objects.count()
        self.pms = list(pms)
        self.vms = list(vms)

    def get_state(self):
        state_matrix = np.zeros((len(self.pms), 3))
        count = 0
        for pm in self.pms:
            deploy_list = Deploy.objects.filter(pm=pm)
            total_cpu = 0
            total_mem = 0
            for d in deploy_list:
                total_cpu += d.vm.cpu
                total_mem += d.vm.memory
            state_matrix[count] = [total_cpu, total_mem, len(list(deploy_list))]
            count += 1
        return np.array(state_matrix.flatten())

    def step(self, action_dqn, action_ppo):
        if action_dqn == 0:
            return self.get_state(), 0.0

        if action_dqn == 1:
            sorted_pm = self.pms
            src_pm_index = int(action_ppo[0] * (self.num_pm - 1))

        if action_dqn == 2:
            sorted_pm = sorted(self.pms, key=lambda pm: get_load(self.develop, pm.pid), reverse=True)
            src_pm_index = int(action_ppo[0] * 5)

        if action_dqn == 3:
            sorted_pm = PhysicalMachine.objects.filter().order_by("-used_cpu")
            src_pm_index = int(action_ppo[0] * 5)

        if action_dqn == 4:
            sorted_pm = PhysicalMachine.objects.filter().order_by("-used_mem")
            src_pm_index = int(action_ppo[0] * 5)

        src_pm = sorted_pm[src_pm_index]
        dst_pm_index = int(action_ppo[2] * (self.num_pm - 1))
        dst_pm = sorted_pm[dst_pm_index]

        length = Deploy.objects.filter(pm=src_pm).count()

        if length == 0:
            return self.get_state(), -0.2

        if dst_pm == src_pm:
            return self.get_state(), -0.2

        vm_index = int(action_ppo[1] * (length - 1))
        deploy_list = list(Deploy.objects.filter(pm=src_pm))
        vm = deploy_list[vm_index].vm

        if not isdeploy(vm, dst_pm):
            return self.get_state(), -0.2

        Deploy.objects.filter(pm=src_pm, vm=vm).delete()
        Deploy.objects.create(
            pm=dst_pm,
            vm=vm,
            message="创建成功",
            method="迁移",
        )

        next_nopms = Reward.nopms()
        next_nom = Reward.nom(src_pm, dst_pm)
        next_ei = Reward.ei()
        next_si = Reward.si()
        reward = Reward.compute_reward(next_nopms - self.nopms, next_nom - self.nom, next_ei - self.ei, next_si - self.si)

        if self.nopms == 0 and self.nom and self.ei == 0 and self.si == 0:
            self.nopms = next_nopms
            self.nom = next_nom
            self.ei = next_ei
            self.si = next_si

            return self.get_state(), 0.0

        self.nopms = next_nopms
        self.nom = next_nom
        self.ei = next_ei
        self.si = next_si

        cpu_load = dst_pm.used_cpu / dst_pm.cpu
        mem_load = dst_pm.used_mem / dst_pm.memory
        if cpu_load >= 0.8:
            reward -= 1
        if mem_load >= 0.8:
            reward -= 1

        return self.get_state(), reward
