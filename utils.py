import numpy as np
import matplotlib.pyplot as plt

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)


def get_load(develop, pm_id):
    return (sum(vm['Cpu'] for vm in develop[pm_id]) + sum(vm['Mem'] for vm in develop[pm_id])) / 2


def print_distribution(pms, env):
    category_counts = {pm['Pid']: {i: 0 for i in range(3)} for pm in pms}
    for pm_id, vm_list in env.develop.items():
        for vm in vm_list:
            category_counts[pm_id][vm['Category']] += 1

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
    plt.title('Number of VMs on Each PM by Category')
    plt.legend()
    plt.show()


def moving_average(a, window_size=9):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

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
