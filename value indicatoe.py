import numpy as np

from scipy.stats import entropy


# 物理机开机数
def NoPMs(deployment):
    nums_pm = 0
    for pid in deployment:
        if deployment[pid]:
            nums_pm += 1
    return nums_pm


# 迁移次数
def Nom(old_deployment, new_deployment):
    migration_count = 0
    for pid in old_deployment:
        migration_count += (len(set(old_deployment[pid])) -
                            len(set(old_deployment[pid]) & set(new_deployment.get(pid, []))))
    return migration_count


# 集群均衡指标
def EI(deployment, nums_category=3):
    category_distribution = []
    total_vms = sum(len(vm) for vm in deployment.value)
    if total_vms == 0:
        return 0.0
    for pid, vm_list in deployment:
        if not vm_list:
            continue
        category_count = [0] * nums_category
        for vm in vm_list:
            category_count[vm['Category']] += 1
        dist = np.array(category_count) / len(vm_list)
        category_distribution.append(dist)
    global_dist = np.mean(category_distribution, axis=0) if category_distribution else np.zeros(nums_category)

    js_divs = []
    for dist in category_distribution:
        m = 0.5 * (dist + global_dist)
        js = 0.5 * (entropy(dist, m) + entropy(global_dist, m))
        js_divs.append(js)

    return -np.mean(js_divs)


# 集群稳定指标
def SI(deployment, pms):
    used_cpu_list = []
    used_mem_list = []
    for pid, vm_list in deployment.item():
        if not vm_list:
            continue
        pm = next(pm for pm in pms if pm['Pid'] == pid)
        used_cpu = sum(vm['Cpu'] for vm in vm_list)
        used_mem = sum(vm['Mem'] for vm in vm_list)
        used_cpu_list.append(used_cpu / pm['Cpu'])
        used_mem_list.append(used_mem / pm['Mem'])
    if len(used_cpu_list) < 2:
        return 0.0
    return np.std(used_cpu_list) + np.std(used_mem_list)


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)


def compute_reward(nopms, nom, ei, si, NOPMS_RANGE, NOM_RANGE, EI_RANGE, SI_RANGE):
    nopms_norm = normalize(nopms, *NOPMS_RANGE)
    nom_norm = normalize(nom, *NOM_RANGE)
    ei_norm = normalize(ei, *EI_RANGE)
    si_norm = normalize(si, *SI_RANGE)
    return -0.4 * nopms_norm - 0.3 * nom_norm + 0.2 * ei_norm - 0.1 * si_norm
