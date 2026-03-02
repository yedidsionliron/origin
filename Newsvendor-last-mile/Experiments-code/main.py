"""
Newsvendor Problem - Last Mile Experiments.

This module implements experimental tests for approximation algorithms
solving the dynamic inventory management problem using various approaches:
- Greedy heuristics
- XGBoost-DP approximation (DPXGBoostApproximator via NewsvendorDPApproximator)
- P50+5 heuristic

The DP solver is provided by the generic DPXGBoostApproximator from dp_lookup.py,
adapted to the newsvendor last-mile state space through NewsvendorDPApproximator.

Typical usage:
    python main.py

Example::

    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    results = runner.run_volume_variability()
    results.to_csv('test_volume_variability.csv', index=False)
"""

import concurrent.futures
import dataclasses
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from dp_lookup import DPXGBoostApproximator
from clp_greedy import greedy_nv_approximation, p50p5
from clp_problem import Problem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Newsvendor-specific DP approximator (wraps DPXGBoostApproximator)
# ============================================================================

class NewsvendorDPApproximator:
    """
    Newsvendor-specific finite-horizon DP approximator built on DPXGBoostApproximator.

    Replaces clp_adp.ADP with a gradient-boosted value function approximation.
    The aggregated state space is (n_ut, n_t, u, ll) — 4-dimensional.
    Time-awareness is injected by setting _current_t before each backward-
    induction step, so the reward function uses the correct period's demand
    distribution without modifying DPXGBoostApproximator.

    Args:
        problem: Newsvendor Problem instance.
        capacity_limit: Maximum inventory capacity (L).
        u_values: Per-location u-factors.
        num_locations: Number of delivery locations (k).
        random_seed: Seed for reproducible state sampling during fit.
    """

    _LARGE_COST = 1e9

    def __init__(
        self,
        problem: Problem,
        capacity_limit: int,
        u_values: List[float],
        num_locations: int,
        random_seed: int = 10,
    ) -> None:
        self.problem = problem
        self.capacity_limit = capacity_limit
        self.u_values = u_values
        self.num_locations = num_locations
        self.random_seed = random_seed

        # Aggregation parameters (replicates clp_adp.ADP)
        self.a_tilde = float(
            np.prod(problem.a[1:problem.k - 1]) ** (1 / (problem.k - 2))
        )
        self._f = 1.1
        self._current_t: int = 0  # updated before each backward-induction step

        # Granularity of 1: approximator evaluates every integer n0, no coarsening needed
        action_space = list(range(1, capacity_limit + 1))
        self._approx = DPXGBoostApproximator(
            horizon=problem.end_of_horizon,
            action_space=action_space,
            reward_fn=self._reward_fn,
            transition_fn=self._transition_fn,
            discount=1.0,  # finite-horizon inventory: no temporal discounting
        )

    def _reward_fn(self, s: np.ndarray, a: int) -> float:
        """Single-period negative cost for state s = (n_ut, n_t, u, ll), action a = n0."""
        n_ut, n_t, u, ll = s
        if a > ll:
            return -self._LARGE_COST
        q = (
            a * self.problem.u[0]
            + n_ut * u
            + n_t * self.problem.u[self.problem.k - 1]
        )
        return -float(self.problem.get_expected_cost_one_period(self._current_t, q))

    def _transition_fn(self, s: np.ndarray, a: int) -> np.ndarray:
        """One-period aggregate-state transition for action a = n0."""
        n_ut, n_t, u, ll = s
        n0 = int(a)
        tenure_prop = (self.a_tilde - 1) / (1 - self.a_tilde ** (-self.problem.k + 2))
        new_n_t = float(round(
            n_t * self.problem.a[self.problem.k - 1]
            + n_ut * self.a_tilde * tenure_prop
        ))
        new_n_ut = float(round(
            n0 * self.problem.a[0]
            + n_ut * self.a_tilde * (1 - tenure_prop)
        ))
        xi = (n0 * self.problem.a[0] / new_n_ut) if new_n_ut > 0 else 1.0
        new_u = round(self.problem.u[1] * xi + u * self._f * (1 - xi), 2)
        return np.array([new_n_ut, new_n_t, new_u, max(ll - n0, 0.0)])

    def fit(self, n_samples: int = 100) -> None:
        """
        Sample representative states and fit one gradient-boosted model per period.

        Replicates DPXGBoostApproximator.fit while injecting time-awareness:
        _current_t is set before each backward-induction step so that reward_fn
        uses the demand distribution of the correct period.

        Args:
            n_samples: Number of state samples for training.
        """
        state_samples = self._sample_states(n_samples)
        approx = self._approx
        N = state_samples.shape[0]
        next_values = np.zeros(N)

        for t in reversed(range(approx.horizon)):
            self._current_t = t
            targets = np.zeros(N)
            for i, s in enumerate(state_samples):
                best = -np.inf
                for a in approx.action_space:
                    r = self._reward_fn(s, a)
                    s_next = self._transition_fn(s, a)
                    val = (
                        r + approx.discount
                        * approx._eval_next(s_next, next_values, state_samples)
                    )
                    if val > best:
                        best = val
                targets[i] = best

            model = GradientBoostingRegressor(**approx.model_params)
            model.fit(state_samples, targets)
            approx.models.insert(0, model)
            next_values = targets.copy()

    def _sample_states(self, n: int) -> np.ndarray:
        """Sample representative (n_ut, n_t, u, ll) states covering the feasible space."""
        L = self.capacity_limit
        u_min = min(self.u_values)
        rng = np.random.RandomState(self.random_seed)
        return np.column_stack([
            rng.uniform(0, L, n),        # n_ut
            rng.uniform(0, L, n),        # n_t
            rng.uniform(u_min, 1.0, n),  # u
            rng.uniform(0, L, n),        # ll
        ])

    def get_policy(
        self,
        n_ut: float,
        n_t: float,
        u: float,
        ll: float,
    ) -> pd.DataFrame:
        """
        Extract a greedy policy by rolling out the fitted value functions.

        At each period t selects the action maximising immediate reward plus
        discounted next-period value predicted by the fitted model.

        Returns:
            DataFrame with columns ['time', 'n0', 'qt', 'tau'], compatible with
            Problem.get_expected_cost().
        """
        approx = self._approx
        rows: List[dict] = []
        s = np.array([n_ut, n_t, u, float(ll)])

        for t in range(approx.horizon):
            self._current_t = t
            ll_curr = s[3]
            best_n0, best_val = 1, -np.inf

            for n0 in approx.action_space:
                if n0 > ll_curr:
                    break
                r = self._reward_fn(s, n0)
                s_next = self._transition_fn(s, n0)
                t_next = t + 1
                v_next = (
                    float(approx.predict(s_next.reshape(1, -1), t_next)[0])
                    if t_next < len(approx.models)
                    else 0.0
                )
                val = r + approx.discount * v_next
                if val > best_val:
                    best_val, best_n0 = val, n0

            n_ut_s, n_t_s, u_s, _ = s
            q = (
                best_n0 * self.problem.u[0]
                + n_ut_s * u_s
                + n_t_s * self.problem.u[self.problem.k - 1]
            )
            rows.append({'time': t, 'n0': best_n0, 'qt': q, 'tau': t + 1})
            s = self._transition_fn(s, best_n0)

        return pd.DataFrame(rows, columns=['time', 'n0', 'qt', 'tau'])


# ============================================================================
# Top-level worker functions for ProcessPoolExecutor
# (must be module-level to be picklable under multiprocessing 'spawn' on Windows)
# ============================================================================

def _worker_volume(
    demands: np.ndarray,
    distribution_type: str,
    num_locations: int,
    capacity_limit: int,
    planning_horizon: int,
    demand_stds: list,
    availability_factors: tuple,
    u_values: list,
    h_threshold: float,
    cost_overage: int,
    cost_underage: int,
    n_ut_initial: int,
    n_t_initial: int,
    u_tilde_initial: float,
    initial_states: tuple,
    random_seed: int,
    v: float,
    i: int,
) -> List[dict]:
    """Run one (volume_factor, iteration) pair; return Greedy + XGB-DP row dicts."""
    problem = Problem(
        distribution_type, num_locations, capacity_limit, planning_horizon,
        demands, demand_stds, availability_factors, u_values,
        h_threshold, cost_overage, cost_underage,
    )
    t0 = time.time()
    greedy_policy = greedy_nv_approximation(problem, initial_states)
    rt_greedy = time.time() - t0
    cost_greedy = problem.get_expected_cost(greedy_policy, initial_states)

    approx = NewsvendorDPApproximator(
        problem, capacity_limit, u_values, num_locations, random_seed
    )
    t0 = time.time()
    approx.fit()
    adp_policy = approx.get_policy(n_ut_initial, n_t_initial, u_tilde_initial, problem.L)
    rt_adp = time.time() - t0
    cost_adp = problem.get_expected_cost(adp_policy, initial_states)

    return [
        {'algorithm': 'Greedy', 'runtime': rt_greedy, 'cost': 1, 'v': v, 'i': i},
        {'algorithm': 'ECA', 'runtime': rt_adp,
         'cost': cost_adp / cost_greedy, 'v': v, 'i': i},
    ]


def _worker_u_variability(
    demands: np.ndarray,
    distribution_type: str,
    num_locations: int,
    capacity_limit: int,
    planning_horizon: int,
    demand_stds: list,
    availability_factors: tuple,
    u_values: list,
    h_threshold: float,
    cost_overage: int,
    cost_underage: int,
    n_ut_initial: int,
    n_t_initial: int,
    u_tilde_initial: float,
    initial_states: tuple,
    random_seed: int,
    u_inc: float,
    i: int,
) -> List[dict]:
    """Run one (u_increment, iteration) pair; return Greedy + XGB-DP rows."""
    problem = Problem(
        distribution_type, num_locations, capacity_limit, planning_horizon,
        demands, demand_stds, availability_factors, u_values,
        h_threshold, cost_overage, cost_underage,
    )
    t0 = time.time()
    greedy_policy = greedy_nv_approximation(problem, initial_states)
    rt_greedy = time.time() - t0
    cost_greedy = problem.get_expected_cost(greedy_policy, initial_states)

    approx = NewsvendorDPApproximator(
        problem, capacity_limit, u_values, num_locations, random_seed
    )
    t0 = time.time()
    approx.fit()
    adp_policy = approx.get_policy(n_ut_initial, n_t_initial, u_tilde_initial, problem.L)
    rt_adp = time.time() - t0
    cost_adp = problem.get_expected_cost(adp_policy, initial_states)

    return [
        {'algorithm': 'Greedy', 'runtime': rt_greedy, 'cost': 1,
         'u_increment': u_inc, 'i': i},
        {'algorithm': 'ECA', 'runtime': rt_adp,
         'cost': cost_adp / cost_greedy, 'u_increment': u_inc, 'i': i},
    ]


def _worker_a_variability(
    demands: np.ndarray,
    distribution_type: str,
    num_locations: int,
    capacity_limit: int,
    planning_horizon: int,
    demand_stds: list,
    availability_factors: tuple,
    u_values: list,
    h_threshold: float,
    cost_overage: int,
    cost_underage: int,
    n_ut_initial: int,
    n_t_initial: int,
    u_tilde_initial: float,
    initial_states: tuple,
    random_seed: int,
    a_range: float,
    i: int,
) -> List[dict]:
    """Run one (availability_range, iteration) pair; return Greedy + XGB-DP rows."""
    problem = Problem(
        distribution_type, num_locations, capacity_limit, planning_horizon,
        demands, demand_stds, availability_factors, u_values,
        h_threshold, cost_overage, cost_underage,
    )
    t0 = time.time()
    greedy_policy = greedy_nv_approximation(problem, initial_states)
    rt_greedy = time.time() - t0
    cost_greedy = problem.get_expected_cost(greedy_policy, initial_states)

    approx = NewsvendorDPApproximator(
        problem, capacity_limit, u_values, num_locations, random_seed
    )
    t0 = time.time()
    approx.fit()
    adp_policy = approx.get_policy(n_ut_initial, n_t_initial, u_tilde_initial, problem.L)
    rt_adp = time.time() - t0
    cost_adp = problem.get_expected_cost(adp_policy, initial_states)

    return [
        {'algorithm': 'Greedy', 'runtime': rt_greedy, 'cost': 1,
         'a_range': a_range, 'i': i},
        {'algorithm': 'ECA', 'runtime': rt_adp,
         'cost': cost_adp / cost_greedy, 'a_range': a_range, 'i': i},
    ]


# ============================================================================
# Configuration
# ============================================================================

@dataclasses.dataclass
class ExperimentConfig:
    """
    Configuration parameters for newsvendor last-mile experiments.

    All derived parameters (demand_stds, u_values, etc.) are computed
    automatically from the base parameters in __post_init__.

    Attributes:
        cost_underage: Per-unit underage penalty.
        cost_overage: Per-unit overage penalty.
        cost_reallocation: Per-unit reallocation cost.
        num_locations: Number of delivery locations (k).
        capacity_limit: Maximum inventory capacity (L).
        planning_horizon: Number of time periods.
        distribution_type: Demand distribution ('normal', 'uniform', or 'gamma').
        demand_means: Mean demand per period.
        coefficient_of_variation: CV used to derive demand standard deviations.
        availability_factors: Per-location availability factors.
        u_increment_factor: Ratio between consecutive u-values.
        initial_states: Initial inventory state tuple.
        random_seed: Seed for reproducible random number generation.
        n_fit_samples: State samples used to fit NewsvendorDPApproximator.
    """
    # Cost parameters
    cost_underage: int = 50
    cost_overage: int = 70
    cost_reallocation: int = 0

    # Problem dimensions
    num_locations: int = 5
    capacity_limit: int = 64
    planning_horizon: int = 10
    distribution_type: str = 'normal'

    # Demand parameters
    demand_means: Tuple = (20, 25, 20, 25, 50, 35, 30, 25, 20, 25)
    coefficient_of_variation: float = 0.14

    # Availability parameters
    availability_factors: Tuple = (0.95, 0.94, 0.93, 0.92, 0.91, 0.9)
    u_increment_factor: float = 1.1

    # Initial state
    initial_states: Tuple = (0, 2, 2, 2, 13)

    # Experiment random seed
    random_seed: int = 10

    # DP approximator fitting
    n_fit_samples: int = 100

    def __post_init__(self) -> None:
        # Vectorised: multiply the whole array at once instead of element-wise loop
        self.demand_stds: List[float] = list(
            np.array(self.demand_means) * self.coefficient_of_variation
        )
        self.u_values: List[float] = [1.0] * self.num_locations
        for i in reversed(range(self.num_locations - 1)):
            self.u_values[i] = self.u_values[i + 1] / self.u_increment_factor

        self.n_ut_initial: int = sum(self.initial_states[1:self.num_locations - 1])
        self.n_t_initial: int = self.initial_states[self.num_locations - 1]
        self.u_tilde_initial: float = (
            sum(self.u_values[1:self.num_locations - 1]) / (self.num_locations - 2)
        )
        self.h_threshold: float = 0.01 * min(self.cost_underage, self.cost_overage)


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """
    Runs experiments comparing greedy and DP-approximation algorithms on the
    newsvendor last-mile problem.

    The DP solver is NewsvendorDPApproximator, which wraps DPXGBoostApproximator
    (dp_lookup.py) with newsvendor-specific reward and transition functions.

    Args:
        config: ExperimentConfig instance with all experiment parameters.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _make_problem(
        self,
        demands: np.ndarray,
        availability_factors: Optional[Tuple] = None,
        u_values: Optional[List[float]] = None,
        capacity_limit: Optional[int] = None,
        cost_underage: Optional[int] = None,
        cost_overage: Optional[int] = None,
    ) -> Problem:
        """Create a Problem instance, using config defaults for unspecified params."""
        cfg = self.config
        return Problem(
            cfg.distribution_type,
            cfg.num_locations,
            capacity_limit if capacity_limit is not None else cfg.capacity_limit,
            cfg.planning_horizon,
            demands,
            cfg.demand_stds,
            availability_factors if availability_factors is not None else cfg.availability_factors,
            u_values if u_values is not None else cfg.u_values,
            cfg.h_threshold,
            cost_overage if cost_overage is not None else cfg.cost_overage,
            cost_underage if cost_underage is not None else cfg.cost_underage,
        )

    def _run_greedy(self, problem: Problem) -> Tuple[float, float]:
        """
        Run the greedy approximation algorithm.

        Returns:
            Tuple of (cost, execution_time).
        """
        start_time = time.time()
        greedy_policy = greedy_nv_approximation(problem, self.config.initial_states)
        execution_time = time.time() - start_time
        cost = problem.get_expected_cost(greedy_policy, self.config.initial_states)
        return cost, execution_time

    def _run_adp(self, problem: Problem) -> Tuple[float, float]:
        """
        Fit NewsvendorDPApproximator and extract a greedy policy.

        Timing covers both fitting and policy extraction, reflecting the full
        cost of producing a solution with the XGBoost-based approximator.

        Returns:
            Tuple of (cost, execution_time).
        """
        cfg = self.config
        approx = NewsvendorDPApproximator(
            problem, cfg.capacity_limit, cfg.u_values,
            cfg.num_locations, cfg.random_seed,
        )
        start_time = time.time()
        approx.fit(cfg.n_fit_samples)
        policy = approx.get_policy(
            cfg.n_ut_initial, cfg.n_t_initial, cfg.u_tilde_initial, problem.L
        )
        execution_time = time.time() - start_time
        cost = problem.get_expected_cost(policy, cfg.initial_states)
        return cost, execution_time

    def _base_worker_kwargs(
        self,
        u_values: Optional[List[float]] = None,
        availability_factors: Optional[Tuple] = None,
    ) -> dict:
        """Build the common keyword args passed to every top-level worker function."""
        cfg = self.config
        return dict(
            distribution_type=cfg.distribution_type,
            num_locations=cfg.num_locations,
            capacity_limit=cfg.capacity_limit,
            planning_horizon=cfg.planning_horizon,
            demand_stds=cfg.demand_stds,
            availability_factors=(
                availability_factors
                if availability_factors is not None
                else cfg.availability_factors
            ),
            u_values=u_values if u_values is not None else cfg.u_values,
            h_threshold=cfg.h_threshold,
            cost_overage=cfg.cost_overage,
            cost_underage=cfg.cost_underage,
            n_ut_initial=cfg.n_ut_initial,
            n_t_initial=cfg.n_t_initial,
            u_tilde_initial=cfg.u_tilde_initial,
            initial_states=cfg.initial_states,
            random_seed=cfg.random_seed,
        )

    # -------------------------------------------------------------------------
    # Public experiment methods
    # -------------------------------------------------------------------------

    def run_volume_variability(
        self,
        mean_demand: float = 30,
        volume_factors: Optional[np.ndarray] = None,
        iterations: int = 5,
    ) -> pd.DataFrame:
        """
        Test algorithm performance across varying demand volume levels.

        Evaluates greedy and XGB-DP (block size 8) under different demand
        variability ratios. Iterations for each factor run in parallel.

        Args:
            mean_demand: Base demand mean.
            volume_factors: Volume multipliers. Defaults to linspace(1, 3, 11).
            iterations: Random repetitions per factor.

        Returns:
            DataFrame with columns ['algorithm', 'runtime', 'cost', 'v', 'i'].
        """
        cfg = self.config
        np.random.seed(cfg.random_seed)
        if volume_factors is None:
            volume_factors = np.linspace(1, 3, 11)

        demand_arrays: Dict[Tuple, np.ndarray] = {
            (v, i): np.random.uniform(mean_demand, mean_demand * v, cfg.planning_horizon)
            for v in volume_factors
            for i in range(iterations)
        }

        base_kwargs = self._base_worker_kwargs()
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for v in volume_factors:
                logger.info(f'Submitting volume factor: {v}')
                for i in range(iterations):
                    futures.append(executor.submit(
                        _worker_volume,
                        demand_arrays[(v, i)],
                        **base_kwargs,
                        v=v,
                        i=i,
                    ))

        rows: List[dict] = []
        for future in futures:
            rows.extend(future.result())

        logger.info('Volume variability test complete')
        return pd.DataFrame(rows, columns=['algorithm', 'runtime', 'cost', 'v', 'i'])

    def run_u_variability(
        self,
        mean_demand: float = 30,
        u_increments: Optional[np.ndarray] = None,
        iterations: int = 5,
    ) -> pd.DataFrame:
        """
        Test algorithm performance across varying availability ratios.

        Iterations for each U-increment factor run in parallel.

        Args:
            mean_demand: Base demand mean.
            u_increments: U-increment factors. Defaults to linspace(1, 2, 6).
            iterations: Random repetitions per increment.

        Returns:
            DataFrame with columns ['algorithm', 'runtime', 'cost', 'u_increment', 'i'].
        """
        cfg = self.config
        np.random.seed(cfg.random_seed)
        if u_increments is None:
            u_increments = np.linspace(1, 2, 6)

        demand_arrays: Dict[Tuple, np.ndarray] = {}
        u_values_map: Dict[float, List[float]] = {}
        for u_inc in u_increments:
            u_vals = [1.0] * cfg.num_locations
            for idx in reversed(range(cfg.num_locations - 1)):
                u_vals[idx] = u_vals[idx + 1] / u_inc
            u_values_map[u_inc] = u_vals
            for i in range(iterations):
                demand_arrays[(u_inc, i)] = np.random.uniform(
                    mean_demand, mean_demand * 2, cfg.planning_horizon
                )

        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for u_inc in u_increments:
                logger.info(f'Submitting U-increment factor: {u_inc}')
                base_kwargs = self._base_worker_kwargs(u_values=u_values_map[u_inc])
                for i in range(iterations):
                    futures.append(executor.submit(
                        _worker_u_variability,
                        demand_arrays[(u_inc, i)],
                        **base_kwargs,
                        u_inc=u_inc,
                        i=i,
                    ))

        rows: List[dict] = []
        for future in futures:
            rows.extend(future.result())

        logger.info('U variability test complete')
        return pd.DataFrame(
            rows, columns=['algorithm', 'runtime', 'cost', 'u_increment', 'i']
        )

    def run_a_variability(
        self,
        mean_demand: float = 30,
        availability_ranges: Optional[np.ndarray] = None,
        iterations: int = 10,
    ) -> pd.DataFrame:
        """
        Test algorithm performance across varying availability ranges.

        Iterations for each range run in parallel.

        Args:
            mean_demand: Base demand mean.
            availability_ranges: Availability range endpoints.
                Defaults to linspace(0.5, 0.95, 11).
            iterations: Random repetitions per range.

        Returns:
            DataFrame with columns ['algorithm', 'runtime', 'cost', 'a_range', 'i'].
        """
        cfg = self.config
        np.random.seed(cfg.random_seed)
        if availability_ranges is None:
            availability_ranges = np.linspace(0.5, 0.95, 11)

        demand_arrays: Dict[Tuple, np.ndarray] = {}
        avail_map: Dict[float, Tuple] = {}
        for a_range in availability_ranges:
            avail_map[a_range] = tuple(np.linspace(1, a_range, cfg.num_locations))
            for i in range(iterations):
                demand_arrays[(a_range, i)] = np.random.uniform(
                    mean_demand, mean_demand * 2, cfg.planning_horizon
                )

        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for a_range in availability_ranges:
                logger.info(f'Submitting availability range: {a_range}')
                base_kwargs = self._base_worker_kwargs(
                    availability_factors=avail_map[a_range]
                )
                for i in range(iterations):
                    futures.append(executor.submit(
                        _worker_a_variability,
                        demand_arrays[(a_range, i)],
                        **base_kwargs,
                        a_range=a_range,
                        i=i,
                    ))

        rows: List[dict] = []
        for future in futures:
            rows.extend(future.result())

        logger.info('Availability range test complete')
        return pd.DataFrame(
            rows, columns=['algorithm', 'runtime', 'cost', 'a_range', 'i']
        )

    def run_amazon_showcase(self) -> None:
        """
        Showcase test using Amazon case study parameters.

        Evaluates P50+5, Greedy, and XGB-DP (block sizes 2/4/8) with realistic
        Amazon routing costs, then sweeps capacity limits.
        """
        cost_underage_amazon = 805   # $230 (shift) * 3.5 (per week) * 1
        cost_overage_amazon = 1750   # $500 (route) * 3.5 (per week)
        cfg = self.config

        problem = self._make_problem(
            np.array(cfg.demand_means, dtype=float),
            cost_underage=cost_underage_amazon,
            cost_overage=cost_overage_amazon,
        )

        # P50+5 heuristic
        logger.info('P50+5 Heuristic:')
        logger.info('=' * 40)
        start_time = time.time()
        p50p5_policy = p50p5(problem, cfg.initial_states)
        runtime = time.time() - start_time
        cost = problem.get_expected_cost(p50p5_policy, cfg.initial_states, 1)
        logger.info(f'Execution time: {runtime:.4f}s')
        logger.info(f'Expected cost: {cost:.2f}\n')

        # Greedy heuristic
        logger.info('Greedy Heuristic:')
        logger.info('=' * 40)
        cost_greedy, runtime = self._run_greedy(problem)
        logger.info(f'Execution time: {runtime:.4f}s')
        logger.info(f'Expected cost: {cost_greedy:.2f}\n')

        # XGB-DP approximator
        logger.info('XGB-DP Approximator:')
        logger.info('=' * 40)
        cost, runtime = self._run_adp(problem)
        logger.info(f'Execution time (fit + rollout): {runtime:.4f}s')
        logger.info(f'Actual cost: {cost:.2f}\n')

        # Varying capacity limits
        logger.info('Varying Capacity Limits:')
        logger.info('=' * 40)
        for limit in [32, 64, 128]:
            logger.info(f'Capacity Limit: {limit}')
            problem_limited = self._make_problem(
                np.array(cfg.demand_means, dtype=float),
                capacity_limit=limit,
                cost_underage=cost_underage_amazon,
                cost_overage=cost_overage_amazon,
            )
            cost, runtime = self._run_adp(problem_limited)
            logger.info(f'  Execution time: {runtime:.4f}s')
            logger.info(f'  Actual cost: {cost:.2f}\n')


if __name__ == '__main__':
    config = ExperimentConfig()
    runner = ExperimentRunner(config)

    results = runner.run_volume_variability()
    logger.info(f'\n{results}')
    results.to_csv('test_volume_variability.csv', index=False)

    # Uncomment to run other experiments:
    # runner.run_u_variability().to_csv('test_u_variability.csv', index=False)
    # runner.run_a_variability().to_csv('test_a_variability.csv', index=False)
    # runner.run_amazon_showcase()
