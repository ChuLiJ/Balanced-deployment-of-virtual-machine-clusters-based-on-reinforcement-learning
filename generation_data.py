import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


pms = generate_pms(50, (8, 64), (32, 256))
vms = generate_vms(200, 64, 256)
vms, centers = cluster_vms(vms, 3)

cpus = [vm['Cpu'] for vm in vms]
mems = [vm['Mem'] for vm in vms]
plt.scatter(cpus, mems, alpha=0.6)
plt.xlabel("CPU Cores")
plt.ylabel("Memory (GB)")
plt.title("Randomly Generated VMs (Before Clustering)")
plt.show()

categories = [vm["Category"] for vm in vms]
plt.scatter(cpus, mems, c=categories, cmap='viridis', alpha=0.6)
plt.xlabel("CPU Cores")
plt.ylabel("Memory (GB)")
plt.title("VMs After K-means Clustering (3 Categories)")
plt.colorbar(label="Category")
plt.show()
