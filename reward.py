import numpy as np

from scipy.stats import entropy

from utils import normalize


class Reward:

    @classmethod
    def nopms(cls, state):
        nums_pm = 0
        for pid in state:
            if state[pid]:
                nums_pm += 1
        return nums_pm

    @classmethod
    def nom(cls, old_state, new_state):
        migration_count = 0
        for pid in old_state:
            migration_count += (len(old_state[pid]) - len(new_state.get(pid, [])))
        return migration_count

    @classmethod
    def ei(cls, state, nums_category=3):
        category_distribution = []
        total_vms = sum(len(vm) for vm in state.values())
        if total_vms == 0:
            return 0.0
        for pid, vm_list in state.items():
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

        return np.mean(js_divs)

    @classmethod
    def si(cls, state, pms):
        used_cpu_list = []
        used_mem_list = []
        for pid, vm_list in state.items():
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

    @classmethod
    def compute_reward(cls, nopms, nom, ei, si):
        # nopms_norm = normalize(cls.nopms(new_state), *NOPMS_RANGE)
        # nom_norm = normalize(cls.nom(old_state, new_state), *NOM_RANGE)
        # ei_norm = normalize(cls.ei(new_state), *EI_RANGE)
        # si_norm = normalize(cls.si(new_state, pms), *SI_RANGE)
        reward = 0.1 * nopms - 0.0 * nom + 10 * ei + 10 * si
        # print(f"Reward breakdown: nopms={nopms_norm}, nom={nom_norm}, ei={ei_norm}, si={si_norm}, total={reward}")
        return reward
