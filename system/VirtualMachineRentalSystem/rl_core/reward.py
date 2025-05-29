import numpy as np

from scipy.stats import entropy
from django.db.models import Count, Prefetch

from ..models import PhysicalMachine, VirtualMachine, Deploy


class Reward:
    @classmethod
    def nopms(cls):
        return Deploy.objects.value('pm').distinct().count()

    @classmethod
    def nom(cls, src_pm, dst_pm):
        pm_list = PhysicalMachine.objects.annotate(deploy_count=Count('deployments'))
        l_dst = 0
        l_src = 0
        for pm in pm_list:
            if pm is src_pm:
                l_src = pm.deploy_count
            elif pm is dst_pm:
                l_dst = pm.deploy_count

        return l_dst - l_src + 2

    @classmethod
    def ei(cls, nums_category=3):
        category_distribution = []
        pm_list = PhysicalMachine.objects.prefetch_related(
            Prefetch(
                'deployments',
                queryset=Deploy.objects.select_related('vm')
            )
        )
        total_vms = VirtualMachine.objects.exclude(category=-1).count()
        if total_vms == 0:
            return 0.0
        for pm in pm_list:
            vm_list = [d.vm for d in pm.deployments.all() if d.vm and d.vm.category != -1]

            if not vm_list:
                continue
            category_count = [0] * nums_category
            for vm in vm_list:
                category_count[vm.category] += 1
            dist = np.array(category_count) / len(vm_list)
            category_distribution.append(dist)

        if not category_distribution:
            return 0.0
        global_dist = np.mean(category_distribution, axis=0)

        js_divs = []
        for dist in category_distribution:
            m = 0.5 * (dist + global_dist)
            js = 0.5 * (entropy(dist, m) + entropy(global_dist, m))
            js_divs.append(js)

        return float(np.mean(js_divs))

    @classmethod
    def si(cls):
        used_cpu_list = []
        used_mem_list = []
        pm_list = PhysicalMachine.objects.all()
        for pm in pm_list:
            deploy_list = Deploy.objects.filter(pm=pm)
            used_cpu = 0
            used_mem = 0
            for d in deploy_list:
                used_cpu += d.vm.cpu
                used_mem += d.vm.memory
            used_cpu_list.append(used_cpu / pm.cpu)
            used_mem_list.append(used_mem / pm.memory)

        if len(used_cpu_list) < 2:
            return 0.0

        return np.std(used_cpu_list) + np.std(used_mem_list)

    @classmethod
    def compute_reward(cls, nopms, nom, ei, si):
        reward = -1 * nopms - 1 * nom + 100 * ei + 100 * si
        return reward
