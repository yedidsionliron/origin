import math
import pandas as pd
from functools import lru_cache
import numpy as np


class ADP:
    def __init__(self, problem, block_size=1):
        self.dp.cache_clear()
        self.p = problem
        self.policy = None
        self.block_size = block_size
        self.a_tilde = np.prod(self.p.a[1:problem.k - 1]) ** (1 / (problem.k - 2))
        # self.a_tilde = sum(self.p.a[1:problem.k - 1]) / (problem.k - 2)
        self.f = 1.1

    @lru_cache(maxsize=None)
    def dp(self, t, n_ut, n_t, u, ll):
        if t >= self.p.end_of_horizon:
            return self.p.get_end_of_horizon_cost(self.generate_ns(0, n_ut, n_t))
        if ll < 0:
            return math.inf, pd.DataFrame(columns=['time', 'n0', 'qt', 'tau'])
        # initializations:
        best_cost, best_tau, best_policy, feasible_n0 = math.inf, math.inf, [], self.get_feasible_n0(ll)
        # main loop:
        for tau in range(t + 1, self.p.end_of_horizon + 1):
            for n0 in feasible_n0:
                q = self.get_q(n0, n_ut, n_t, u)
                n_t_l, n_ut_l, u_l = self.update_ns_u_multi_period(tau - t + 1, n0, n_ut, n_t, u)
                ctg, ptg = self.dp(tau, n_ut_l, n_t_l, u_l, ll - n0)
                cost = self.get_expected_cost_window(t, tau, n0, n_ut, n_t, u) + ctg
                if cost < best_cost:
                    best_cost, best_tau = cost, tau
                    best_policy = pd.concat([ptg, pd.DataFrame(
                        data={'time': t, 'n0': n0, 'qt': q, 'tau': tau, 'ctg': ctg, 'cost': cost}, index=[0])])
        return best_cost, best_policy

    def get_feasible_n0(self, ll):
        return range(self.block_size, ll + 1, self.block_size)

    def update_ns_u(self, n0, n_ut, n_t, u):
        tenure_proportion = self.calc_tenure_proportion()
        n_t = float(round(n_t * self.p.a[self.p.k - 1] + n_ut * self.a_tilde * tenure_proportion))
        n_ut = float(round((n0 * self.p.a[0]) + (n_ut * self.a_tilde * (1 - tenure_proportion))))
        if n_ut == 0:
            xi = 1
        else:
            xi = n0 * self.p.a[0] / n_ut
        u = round(self.p.u[1] * xi + u * self.f * (1 - xi), 2)
        return n_t, n_ut, u

    def calc_tenure_proportion(self):
        # return (self.p.k-3)/(self.p.k-2)
        return (self.a_tilde - 1) / (1 - self.a_tilde ** (-self.p.k + 2))

    def update_ns_u_multi_period(self, tau, n0, n_ut, n_t, u):
        for t in range(1, tau):
            n_t, n_ut, u = self.update_ns_u(n0, n_ut, n_t, u)
            n0 = 0
        return n_t, n_ut, u

    def get_q(self, n0, n_ut, n_t, u):
        return n0 * self.p.u[0] + n_ut * u + n_t * self.p.u[self.p.k - 1]

    def generate_ns(self, n0, n_ut, n_t):
        ns = [0] * self.p.k
        ns[0] = n0
        for k in range(1, self.p.k - 2):
            ns[k] = n_ut / (self.p.k - 2)
        ns[self.p.k - 1] = n_t
        return tuple(ns)

    def get_expected_cost_window(self, t, tau, n0, n_ut, n_t, u, verbose=0):
        cost = 0
        for t_ in range(t, tau):
            q = self.get_q(n0, n_ut, n_t, u)
            current_cost = self.p.get_expected_cost_one_period(t_, q)
            cost += current_cost
            if verbose:
                print('time: ', t_, 'n0: ', n0, 'n_ut: ', n_ut, 'n_t: ', n_t, 'q: ', round(q, 2), 'mu: ',
                      self.p.mus[t_], 'cost: ',
                      round(current_cost, 2))
            n_t, n_ut, u = self.update_ns_u(n0, n_ut, n_t, u)
            n0 = 0
        return cost + self.p.H

    def get_expected_cost(self, policy, n_ut, n_t, u, verbose=0):
        policy = policy.sort_values(by=['time'])
        if policy['time'].iloc[0] > 0:
            policy = pd.concat([pd.DataFrame(data={'time': 0, 'n0': 0, 'qt': 0}, index=[0]), policy])
        cost = 0
        policy['tau'] = policy.time.shift(-1)
        policy.iloc[-1, policy.columns.get_loc('tau')] = self.p.end_of_horizon
        for index, row in policy.iterrows():
            t = int(row['time'])
            tau = int(row['tau'])
            n0 = row['n0']
            cost += self.get_expected_cost_window(t, tau, n0, n_ut, n_t, u, verbose)
            n_t, n_ut, u = self.update_ns_u_multi_period(tau - t + 1, n0, n_ut, n_t, u)
        return cost
