import copy
import numpy as np

from reward import Reward
from utils import isdeploy, get_load


def greedy_migration(pms, develop):
    src_pm = max(pms, key=lambda pm: get_load(develop, pm['Pid']))
    sorted_pm = sorted(pms, key=lambda pm: get_load(develop, pm['Pid']))
    vm = np.random.choice(develop[src_pm['Pid']])
    dst_id = 0
    dst_pm = sorted_pm[dst_id]

    if src_pm == dst_pm:
        return develop, -1.0

    if not isdeploy(vm, dst_pm, develop):
        return develop, -0.5

    next_develop = copy.deepcopy(develop)
    next_develop[src_pm['Pid']].remove(vm)
    next_develop[dst_pm['Pid']].append(vm)

    return next_develop, Reward.compute_reward(develop, next_develop, pms, (0, 50), (0, 200), (-1.0, 0.0), (0.0, 2.0))
