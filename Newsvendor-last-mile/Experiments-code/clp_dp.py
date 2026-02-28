import math
import pandas as pd
from functools import lru_cache


class DP:
    def __init__(self, problem, block_size=1):
        self.p = problem
        self.policy = None
        self.block_size = block_size

    @lru_cache(maxsize=None)
    def dp(self, t, ns, ll):
        if t >= self.p.end_of_horizon:
            return self.p.get_end_of_horizon_cost(ns)
        if ll < 0:
            return math.inf, pd.DataFrame(columns=['time', 'n0', 'qt', 'tau'])
        # initializations:
        best_cost, best_tau, best_policy, lns, feasible_n0 = math.inf, math.inf, [], list(ns), self.get_feasible_n0(
            ll)

        # main loop:
        for tau in range(t + 1, self.p.end_of_horizon + 1):
            for n0 in feasible_n0:
                lns[0] = n0
                qt = self.p.get_q(lns)
                new_ns = tuple(self.update_ns_multi_period(lns, tau - t + 1))
                ctg, ptg = self.dp(tau, new_ns, ll - n0)
                cost = self.p.get_expected_cost_window(t, tau, tuple(lns)) + ctg
                if cost < best_cost:
                    best_cost, best_tau = cost, tau
                    best_policy = pd.concat(
                        [ptg, pd.DataFrame(data={'time': t, 'n0': n0, 'qt': qt, 'tau': tau}, index=[0])])
        return best_cost, best_policy

    def get_feasible_n0(self, ll):
        return range(self.block_size, ll + 1, self.block_size)

    def update_ns(self, ns):
        return self.p.update_ns(ns)

    def update_ns_multi_period(self, ns, tau):
        for t in range(1, tau):
            ns = self.update_ns(ns)
        return ns