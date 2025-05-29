import copy
import numpy as np

from reward import Reward
from utils import isdeploy, get_load, get_cpu_load, get_mem_load


def greedy_migration(pms, develop, nopms, nom, ei, si):
    src_pm = max(pms, key=lambda pm: get_load(develop, pm['Pid']))
    sorted_pm = sorted(pms, key=lambda pm: get_load(develop, pm['Pid']))
    vm = np.random.choice(develop[src_pm['Pid']])
    dst_id = 0
    dst_pm = sorted_pm[dst_id]

    if src_pm == dst_pm:
        return develop, -0.2, nopms, nom, ei, si

    if not isdeploy(vm, dst_pm, develop):
        return develop, -0.2, nopms, nom, ei, si

    next_develop = copy.deepcopy(develop)
    next_develop[src_pm['Pid']].remove(vm)
    next_develop[dst_pm['Pid']].append(vm)
    next_nopms = Reward.nopms(next_develop)
    next_nom = Reward.nom(develop, src_pm['Pid'], dst_pm['Pid'])
    next_ei = Reward.ei(next_develop)
    next_si = Reward.si(next_develop, pms)
    reward = Reward.compute_reward(next_nopms - nopms, next_nom - nom, next_ei - ei, next_si - si)

    if nopms == 0 and nom and ei == 0 and si == 0:
        develop = next_develop
        nopms = next_nopms
        nom = next_nom
        ei = next_ei
        si = next_si

        return next_develop, 0.0, next_nopms, next_nom, next_ei, next_si

    cpu_load = get_cpu_load(develop, dst_pm['Pid']) / pms[dst_pm['Pid']]['Cpu']
    mem_load = get_mem_load(develop, dst_pm['Pid']) / pms[dst_pm['Pid']]['Mem']

    if cpu_load >= 0.8:
        reward -= 1
    if mem_load >= 0.8:
        reward -= 1

    return next_develop, reward, next_nopms, next_nom, next_ei, next_si
