import math
import scipy.stats
import pandas as pd
from functools import lru_cache


class Problem:
    # Cost problem includes all problem parameters, as well as general problem functions that are not solution specific
    def __init__(self, distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u, eoh_buffer=3):
        self.get_end_of_horizon_cost.cache_clear()
        self.get_expected_cost_one_period.cache_clear()
        self.a = a  # vector length k
        self.mus = mus  # vector of mus
        self.sigmas = sigmas  # vector of sigmas
        self.end_of_horizon = end_of_horizon  # number
        self.u = u  # vector, length k
        self.k = k  # number of nursery types
        self.L = L
        self.H = H
        self.distribution = distribution
        self.c_o = c_o
        self.c_u = c_u
        self.mus = self.mus
        self.sigmas = self.sigmas

        self.eoh_buffer = eoh_buffer
        self.mean_mu = sum(mus) / len(mus)
        self.mus_eoh = tuple([self.mean_mu] * self.eoh_buffer)
        self.mean_sigma = self.mean_mu * sum([sigmas[i] / mus[i] for i in range(len(mus))]) / len(mus)
        self.sigmas_eoh = tuple([self.mean_sigma] * self.eoh_buffer)
        self.initial_ns = [self.get_initial_values(self.mean_mu, self.mean_sigma)] * k
        self.final_ns = [self.get_initial_values(self.mean_mu, self.mean_sigma)] * k


        self.mus = tuple(list(self.mus) + list(self.mus_eoh))
        self.sigmas = tuple(list(self.sigmas) + list(self.sigmas_eoh))

    def get_initial_values(self, mu, sigma):
        critical_ratio = self.c_u / (self.c_o + self.c_u)
        Q = self.ppf(critical_ratio, mu, sigma)
        denominator = self.u[0] + self.u[1] * self.a[0] + self.u[2] * self.a[0] * self.a[1] + self.u[3] * self.a[0] * \
                      self.a[1] * self.a[2] * 1 / (1 - self.a[3])
        n0 = Q / denominator
        return n0

    @lru_cache(maxsize=None)
    def get_end_of_horizon_cost(self, ns):
        cost = 0
        policy = pd.DataFrame(columns=['time', 'n0', 'qt', 'tau'])
        ns = list(ns)
        critical_ratio = self.c_u / (self.c_o + self.c_u)
        for t in range(self.eoh_buffer):
            q_target = self.ppf(critical_ratio, self.mus[t + self.end_of_horizon], self.sigmas[t + self.end_of_horizon])
            q_current = self.get_q(tuple(ns))
            n0 = max(math.ceil((q_target - q_current) / self.u[0]), 0)
            ns[0] = n0
            q = self.get_q(tuple(ns))
            cost += self.get_expected_cost_one_period(t + self.end_of_horizon, q)
            policy = pd.concat([policy, pd.DataFrame(
                data={'time': t + self.end_of_horizon, 'n0': n0, 'qt': q, 'tau': t + self.end_of_horizon + 1},
                index=[0])])
            ns = list(self.update_ns(tuple(ns)))
        return cost, policy

    def ppf(self, x, mu, sigma):
        if self.distribution == 'uniform':
            a, b = mu - (2 ** 0.5) * sigma, mu + (2 ** 0.5) * sigma
            return round(scipy.stats.uniform.ppf(x, a, b - a))
        if self.distribution == 'normal':
            return scipy.stats.norm.ppf(x, loc=mu, scale=sigma)
        if self.distribution == 'gamma':
            alpha = (mu ** 2) / (sigma ** 2)
            beta = mu / (sigma ** 2)
            scale = 1 / beta
            return round(scipy.stats.gamma.ppf(x, alpha, scale=scale))

    def get_q(self, ns):
        q = 0
        for i in range(self.k):
            q += ns[i] * self.u[i]
        return q

    def update_ns(self, ns):
        ns = list(ns)
        ns[self.k - 1] = ns[self.k - 1] * self.a[self.k - 1] + ns[self.k - 2] * self.a[self.k - 2]
        for i in reversed(range(self.k - 2)):
            ns[i + 1] = ns[i] * self.a[i]
        ns[0] = 0
        return ns

    def update_ns_multi_period(self, ns, tau):
        for t in range(1, tau):
            ns = self.update_ns(ns)
        return ns

    @lru_cache(maxsize=None)
    def get_expected_cost_one_period(self, t, q):
        if self.distribution == 'uniform':
            return self.get_expected_cost_one_period_uniform(t, q)
        if self.distribution == 'normal':
            return self.get_expected_cost_one_period_normal(t, q)
        if self.distribution == 'gamma':
            return self.get_expected_cost_one_period_gamma(t, q)

    def get_expected_cost_one_period_uniform(self, t, q):
        a, b = self.mus[t] - (2 ** 0.5) * self.sigmas[t], self.mus[t] + (2 ** 0.5) * self.sigmas[t]
        q = min(max(q, a), b)
        integral_o, integral_u = ((a - q) ** 2) / (2 * (b - a)), ((b - q) ** 2) / (2 * (b - a))
        return (self.c_o * integral_o) + (self.c_u * integral_u)

    def get_expected_cost_one_period_normal(self, t, q):
        integral_o = scipy.integrate.quad(
            lambda x: (q - x) / (self.sigmas[t] * (2 * math.pi) ** 0.5) * math.exp(
                (x - self.mus[t]) ** 2 / (-2 * self.sigmas[t] ** 2)), 0, q,
            points=[self.mus[t] - 5 * self.sigmas[t], self.mus[t] + 5 * self.sigmas[t]])
        integral_u = scipy.integrate.quad(
            lambda x: (x - q) / (self.sigmas[t] * (2 * math.pi) ** 0.5) * math.exp(
                (x - self.mus[t]) ** 2 / (-2 * self.sigmas[t] ** 2)), q, 1e10,
            points=[self.mus[t] - 5 * self.sigmas[t], self.mus[t] + 5 * self.sigmas[t]])
        return (self.c_o * integral_o[0]) + (self.c_u * integral_u[0])

    def get_expected_cost_one_period_gamma(self, t, q):
        alpha = (self.mus[t] ** 2) / (self.sigmas[t] ** 2)
        beta = self.mus[t] / (self.sigmas[t] ** 2)
        integral_o = scipy.integrate.quad(
            lambda x: (q - x) * (x ** (alpha - 1) * math.exp(-beta * x) * (beta ** alpha)) / (
                scipy.special.gamma(alpha, out=None)), 0, q)
        integral_u = scipy.integrate.quad(
            lambda x: (x - q) * (x ** (alpha - 1) * math.exp(-beta * x) * (beta ** alpha)) / (
                scipy.special.gamma(alpha, out=None)), q, math.inf)
        return (self.c_o * integral_o[0]) + (self.c_u * integral_u[0])

    def get_expected_cost_window(self, t, tau, ns, verbose=0):
        cost = 0
        for t_ in range(t, tau):
            q = self.get_q(ns)
            current_cost = self.get_expected_cost_one_period(t_, q)
            cost += current_cost
            if verbose:
                print('time: ', t_, 'n0: ', ns, 'q: ', round(q, 2), 'mu: ', self.mus[t_], 'cost: ',
                      round(current_cost, 2))
            ns = self.update_ns(ns)
        return cost + self.H

    def get_expected_cost(self, policy, ns, verbose=0):
        policy = policy.sort_values(by=['time'])
        if policy['time'].iloc[0] > 0:
            policy = pd.concat([pd.DataFrame(data={'time': 0, 'n0': 0, 'qt': 0}, index=[0]), policy])
        ns = list(ns)
        cost = 0
        policy['tau'] = policy.time.shift(-1)
        policy.iloc[-1, policy.columns.get_loc('tau')] = self.end_of_horizon
        for index, row in policy.iterrows():
            t = int(row['time'])
            tau = int(row['tau'])
            n0 = row['n0']
            ns[0] = n0
            cost += self.get_expected_cost_window(t, tau, tuple(ns), verbose)
            ns = self.update_ns_multi_period(ns, tau - t + 1)
        return cost
