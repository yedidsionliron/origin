import pandas as pd
import time
import numpy as np

from clp_adp import ADP
from clp_greedy import greedy_nv_approximation, p50p5
from clp_problem import Problem

import random


c_u = 50
c_o = 70
c_ru = 0
n_initial = 180
distribution = 'normal'  # use uniform/ normal/ gamma
k = 5
L = 64
end_of_horizon = 10
mus = (20, 25, 20, 25, 50, 35, 30, 25, 20, 25)
cv = 0.14
sigmas = [mu * cv for mu in mus]
a = (0.95, 0.94, 0.93, 0.92, 0.91, 0.9)
u_increments = 1.1
u = [1] * k
for i in reversed(range(k - 1)):
    u[i] = u[i + 1] / u_increments
initial_ns = tuple([0, 2, 2, 2, 13])
n_ut, n_t, u_tilde = sum(initial_ns[1:k - 1]), initial_ns[k - 1], sum(u[1:k - 1]) / (k - 2)
H = 0.01 * min(c_u, c_o)
mean_mu = 30

def test_volume_variability():
    print('test_volume_variability')
    random.seed(10)
    results = pd.DataFrame(columns=['algorithm', 'runtime', 'cost', 'v', 'i'])
    vs = np.linspace(1, 4, 11)
    for v in vs:
        print('v:',v)
        for i in range(10):
            print('i:',i)
            ADP.dp.cache_clear()
            Problem.get_expected_cost_one_period.cache_clear()
            mus = np.random.uniform(mean_mu, mean_mu * v, end_of_horizon)
            p1 = Problem(distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)

            start_time = time.time()
            greedy_policy = greedy_nv_approximation(p1, initial_ns)
            execution_time = time.time() - start_time
            cost_greedy = p1.get_expected_cost(greedy_policy, initial_ns)
            results.loc[len(results)] = {'algorithm': 'Greedy', 'runtime': execution_time, 'cost': 1, 'v': v, 'i': i}

            for j in range(2, 4):
                print('j:',j)
                b = 2 ** j
                dp_app = ADP(p1, b)
                start_time = time.time()
                dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
                execution_time = time.time() - start_time
                cost = p1.get_expected_cost(dp_app_policy, initial_ns)
                results.loc[len(results)] = {'algorithm': 'ECA_' + str(b), 'runtime': execution_time,
                                             'cost': cost / cost_greedy, 'v': v, 'i': i}

    print(results)
    results.to_csv('test_volume_variability.csv', index=False)


def test_block_size():
    print('test_block_size')
    random.seed(10)
    results = pd.DataFrame(columns=['algorithm', 'runtime', 'cost', 'B','j'])
    for j in range(10):
        print('j:', j)
        ADP.dp.cache_clear()
        Problem.get_expected_cost_one_period.cache_clear()
        mus = np.random.uniform(mean_mu, mean_mu * 2, end_of_horizon)
        p1 = Problem(distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)
        start_time = time.time()
        greedy_policy = greedy_nv_approximation(p1, initial_ns)
        execution_time = time.time() - start_time
        cost_greedy = p1.get_expected_cost(greedy_policy, initial_ns)
        results.loc[len(results)] = {'algorithm': 'Greedy', 'runtime': execution_time, 'cost': 1, 'B': 'nan','j':j}
        for i in range(1, 7):
            print('i:', i)
            b = 2 ** i
            dp_app = ADP(p1, b)
            start_time = time.time()
            dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
            execution_time = time.time() - start_time
            cost = p1.get_expected_cost(dp_app_policy, initial_ns)
            results.loc[len(results)] = {'algorithm': 'ECA', 'runtime': execution_time, 'cost': cost / cost_greedy, 'B': b,'j':j}

    print(results)
    results.to_csv('test_block_size.csv', index=False)


def test_u_variability():
    print('test_u_variability')
    random.seed(10)
    results = pd.DataFrame(columns=['algorithm', 'runtime', 'cost', 'u_increment', 'i'])
    u_increments = np.linspace(1, 2, 11)
    for u_increment in u_increments:
        print('u_increment:', u_increment)
        u = [1] * k
        for i in reversed(range(k - 1)):
            u[i] = u[i + 1] / u_increment

        for i in range(50):
            print('i:',i)
            mus = np.random.uniform(mean_mu, mean_mu * 2, end_of_horizon)
            p1 = Problem(distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)

            start_time = time.time()
            greedy_policy = greedy_nv_approximation(p1, initial_ns)
            execution_time = time.time() - start_time
            cost_greedy = p1.get_expected_cost(greedy_policy, initial_ns)
            results.loc[len(results)] = {'algorithm': 'Greedy', 'runtime': execution_time, 'cost': 1,
                                         'u_increment': u_increment, 'i': i}

            for j in range(3, 4):
                print('j:',j)
                b = 2 ** j
                dp_app = ADP(p1, b)
                start_time = time.time()
                dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
                execution_time = time.time() - start_time
                cost = p1.get_expected_cost(dp_app_policy, initial_ns)
                results.loc[len(results)] = {'algorithm': 'ECA_' + str(b), 'runtime': execution_time,
                                             'cost': cost / cost_greedy, 'u_increment': u_increment, 'i': i}
    print(results)
    results.to_csv('test_u_variability.csv', index=False)


def test_a_variability():
    print('test_a_variability')
    random.seed(10)
    results = pd.DataFrame(columns=['algorithm', 'runtime', 'cost', 'a_range', 'i'])
    a_ranges = np.linspace(0.5, 0.95, 11)
    for a_range in a_ranges:
        print('a_range', a_range)
        a = list(np.linspace(1,a_range, k))
        for i in range(10):
            print('i:', i)
            mus = np.random.uniform(mean_mu, mean_mu * 2, end_of_horizon)
            p1 = Problem(distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)

            start_time = time.time()
            greedy_policy = greedy_nv_approximation(p1, initial_ns)
            execution_time = time.time() - start_time
            cost_greedy = p1.get_expected_cost(greedy_policy, initial_ns)
            results.loc[len(results)] = {'algorithm': 'Greedy', 'runtime': execution_time, 'cost': 1,
                                         'a_range': a_range, 'i': i}

            for j in range(2, 4):
                print('j:', j)
                b = 2 ** j
                dp_app = ADP(p1, b)
                start_time = time.time()
                dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
                execution_time = time.time() - start_time
                cost = p1.get_expected_cost(dp_app_policy, initial_ns)
                results.loc[len(results)] = {'algorithm': 'ECA_' + str(b), 'runtime': execution_time,
                                             'cost': cost / cost_greedy, 'a_range': a_range, 'i': i}
    print(results)
    results.to_csv('test_a_variability.csv', index=False)


def test_amazon_showcase():
    c_u = 805  # 230 (spr) * 3.5 (shifts per week) * 1 (per 3rd p marginal cost)
    c_o = 1750  # 500 (cost of a single rout) * 3.5 (times per week)
    mus = (20, 25, 20, 25, 50, 35, 30, 25, 20, 25)
    p1 = Problem(distribution, k, L, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)

    print('P50+5:\n=======')
    start_time = time.time()
    p50p5_policy = p50p5(p1, initial_ns)
    print('execution time is ', time.time() - start_time)
    print(p1.get_expected_cost(p50p5_policy, initial_ns, 1))

    print('Greedy:\n=======')
    start_time = time.time()
    greedy_policy = greedy_nv_approximation(p1, initial_ns)
    print('execution time is ', time.time() - start_time)
    print(p1.get_expected_cost(greedy_policy, initial_ns, 1))

    # TEST APPROXIMATION WITH DIFFERENT BLOCK SIZE:
    n_ut, n_t, u_tilde = sum(initial_ns[1:p1.k - 1]), initial_ns[p1.k - 1], sum(u[1:p1.k - 1]) / (p1.k - 2)

    print('Approx DP w. 8:\n==========')
    dp_app = ADP(p1, 8)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    print('Approx DP w. 4:\n==========')
    dp_app = ADP(p1, 4)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    print('Approx DP w. 2:\n==========')
    dp_app = ADP(p1, 2)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    # TEST RUNTIME FOR DIFFERENT LIMITS:
    p1 = Problem(distribution, k, 32, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)
    print('Limit 32:\n==========')
    dp_app = ADP(p1, 8)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    p1 = Problem(distribution, k, 64, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)
    print('Limit 64:\n==========')
    dp_app = ADP(p1, 8)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    p1 = Problem(distribution, k, 128, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)
    print('Limit 128:\n==========')
    dp_app = ADP(p1, 8)
    start_time = time.time()
    dp_app_cost, dp_app_policy = dp_app.dp(0, n_ut, n_t, u_tilde, p1.L)
    print('execution time is ', time.time() - start_time)
    print(dp_app.get_expected_cost(dp_app_policy, n_ut, n_t, u_tilde, 1))
    print('Approximated cost is: ', dp_app_cost)
    print(p1.get_expected_cost(dp_app_policy, initial_ns, 1))

    # print('Optimal DP:\n===========')
    # dp_opt = DP(p1)
    # start_time = time.time()
    # dp_opt_cost, dp_opt_policy = dp_opt.dp(0, initial_ns, p1.L)
    # print('execution time is ', time.time() - start_time)
    # print(p1.get_expected_cost(dp_opt_policy, initial_ns, 1))


if __name__ == '__main__':
    test_u_variability()
    
