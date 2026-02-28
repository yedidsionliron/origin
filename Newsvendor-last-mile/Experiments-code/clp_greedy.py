import math
import pandas as pd


# def nv(mu, sigma):
#    if distribution == 'uniform':
#        a = mu - (2 ** 0.5) * sigma
#        b = mu + (2 ** 0.5) * sigma
#        return round(scipy.stats.uniform.ppf(c_u / (c_o + c_u), a, b - a))
#    if distribution == 'normal':
#        return round(mu + sigma * scipy.stats.norm.ppf(c_u / (c_o + c_u)))
#    if distribution == 'gamma':
#        alpha = (mu ** 2) / (sigma ** 2)
#        beta = mu / (sigma ** 2)
#        scale = 1 / beta
#        return round(scipy.stats.gamma.ppf(c_u / (c_o + c_u), alpha, scale=scale))


# functions for the greedy algorithm:
def ternary_search(f, ll, r, epsilon):
    m1 = ll + ((r - ll) / 3)
    m2 = r - ((r - ll) / 3)
    if abs(f(m1) - f(m2)) < epsilon:
        return (m1 + m2) / 2
    if f(m1) < f(m2):
        return ternary_search(f, ll, m2, epsilon)
    if f(m1) > f(m2):
        return ternary_search(f, m1, r, epsilon)


def get_nq_for_window(p, t, tau, ll, ns):
    lb, ub = 0, ll
    ns = list(ns)

    def f(x): return update_n_return_expected_cost_window(p, t, tau, ns, x)

    n0 = math.ceil(ternary_search(f, lb, ub, 0.5))
    ns[0] = n0
    q = p.get_q(ns)
    cost = p.get_expected_cost_window(t, tau, tuple(ns))
    return cost, q, n0


def get_optimal_nq_for_window(p, t, tau, ll, ns):
    lb, ub = 0, ll + 1
    ns = list(ns)
    best_cost, best_n0, best_q = math.inf, math.inf, math.inf
    for n0 in range(lb, ub):
        ns[0] = n0
        q = p.get_q(ns)
        cost = p.get_expected_cost_window(t, tau, tuple(ns))
        if t + tau == p.end_of_horizon - 1:
            eoh_cost, eoh_policy = p.get_end_of_horizon_cost(tuple(ns))
            cost += eoh_cost
        if cost < best_cost:
            best_cost = cost
            best_n0 = n0
            best_q = q
    return best_cost, best_q, best_n0


def update_n_return_expected_cost_window(p, t, tau, ns, n0):
    ns = list(ns)
    ns[0] = n0
    return p.get_expected_cost_window(t, tau, tuple(ns))


def greedy_nv_approximation(p, ns):
    ns = list(ns)
    t, ll = 0, p.L
    policy = pd.DataFrame(columns=['time', 'n0', 'qt'])
    while t < p.end_of_horizon and ll > 0:
        best_tau, best_cost, best_q, best_n = math.inf, math.inf, math.inf, math.inf
        for tau in range(t + 1, p.end_of_horizon + 1):
            cost_per_window, q, n0 = get_optimal_nq_for_window(p, t, tau, ll, ns)
            cost_per_period = cost_per_window / (tau - t)
            if t + tau == p.end_of_horizon - 1:
                cost_per_period = cost_per_window / (tau - t + p.eoh_buffer)
            if cost_per_period < best_cost:
                best_tau, best_cost, best_q, best_n = tau, cost_per_period, q, n0
        policy.loc[len(policy)] = {'time': t, 'n0': best_n, 'qt': best_q}
        ll = ll - best_n
        ns[0] = best_n
        ns = p.update_ns_multi_period(tuple(ns), best_tau - t + 1)
        t = best_tau
    cost, eoh_policy = p.get_end_of_horizon_cost(tuple(ns))
    policy = pd.concat([policy, eoh_policy])
    return policy


def p50p5(p, ns):
    ns = list(ns)
    policy = pd.DataFrame(columns=['time', 'n0', 'qt'])
    targets = [mu * 1.05 for mu in p.mus]
    for t in range(p.end_of_horizon):
        q = p.get_q(ns)
        while q < targets[t]:
            ns[0] += 1
            q = p.get_q(ns)
        policy.loc[len(policy)] = {'time': t, 'n0': ns[0], 'qt': q}
        ns = p.update_ns(ns)
    if sum(policy['n0']) > p.L:
        policy['n0'] = [round(x * p.L / sum(policy['n0'])) for x in policy['n0']]
    cost, eoh_policy = p.get_end_of_horizon_cost(tuple(ns))
    policy = pd.concat([policy, eoh_policy])
    return policy
