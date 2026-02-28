import pandas as pd
import time
import numpy as np

from clp_adp import ADP
from clp_greedy import greedy_nv_approximation, p50p5
from clp_problem import Problem

import random

distribution = 'normal'  # use uniform/ normal/ gamma

mean_mu = 30


def test_amazon_showcase():
    # problem parameters:
    k = 4
    a = (0.9, 0.93, 0.92, 0.97)
    u = (0.7, 0.8, 0.9, 1)
    c_u = 805  # 230 (spr) * 3.5 (shifts per week) * 1 (per 3rd p marginal cost)
    c_o = 1750  # 500 (cost of a single rout) * 3.5 (times per week)    ##370*3.5
    mus = (32, 56, 58, 44, 46, 62, 64, 66, 67, 66, 55, 66, 52)
    end_of_horizon = len(mus)
    cv = 0.138
    sigmas = [mu * cv for mu in mus]
    H = 0.01 * min(c_u, c_o)
    limits = [128, 256]

    # algorithm initialization:
    mean_mus = mus[0]
    initial_n = mean_mus / sum(u)
    initial_ns = tuple([initial_n, initial_n, initial_n, initial_n])

    for limit in limits:
        p1 = Problem(distribution, k, limit, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u)
        print('Limit: ', limit, '\n=========\n=========')
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


def test_eoh():
    # problem parameters:
    k = 4
    a = (0.9, 0.93, 0.92, 0.97)
    u = (0.7, 0.8, 0.9, 1)
    c_u = 805  # 230 (spr) * 3.5 (shifts per week) * 1 (per 3rd p marginal cost)
    c_o = 1750  # 500 (cost of a single rout) * 3.5 (times per week)
    mus = (32, 56, 58, 44, 46, 62, 64, 66, 67, 66, 55, 66, 52)
    end_of_horizon = len(mus)
    cv = 0.138
    sigmas = [mu * cv for mu in mus]
    H = 0.01 * min(c_u, c_o)
    limits = [128, 256]
    eohs = [2,3,4,5,6,7,8,9,10]

    # algorithm initialization:
    mean_mus = mus[0]
    initial_n = mean_mus / sum(u)
    initial_ns = tuple([initial_n, initial_n, initial_n, initial_n])
    for eoh in eohs:
        p1 = Problem(distribution, k, 128, end_of_horizon, mus, sigmas, a, u, H, c_o, c_u, eoh)
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

if __name__ == '__main__':
    test_eoh()
    #test_amazon_showcase()
