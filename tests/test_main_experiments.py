"""
Unit tests for ExperimentConfig and ExperimentRunner in main.py.

Run with:
    cd "Newsvendor-last-mile/Experiments-code"
    pytest ../../tests/test_main_experiments.py -v
"""

import sys
import os

import pytest

# Add the experiments source directory to path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'Newsvendor-last-mile', 'Experiments-code'),
)

from main import ExperimentConfig, ExperimentRunner


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass and derived parameter computation."""

    def test_defaults_are_set(self):
        cfg = ExperimentConfig()
        assert cfg.cost_underage == 50
        assert cfg.cost_overage == 70
        assert cfg.num_locations == 5
        assert cfg.capacity_limit == 64
        assert cfg.planning_horizon == 10
        assert cfg.random_seed == 10

    def test_demand_stds_length_matches_means(self):
        cfg = ExperimentConfig()
        assert len(cfg.demand_stds) == len(cfg.demand_means)

    def test_demand_stds_scale_by_cv(self):
        cfg = ExperimentConfig()
        for mu, sigma in zip(cfg.demand_means, cfg.demand_stds):
            assert sigma == pytest.approx(mu * cfg.coefficient_of_variation)

    def test_u_values_length_matches_num_locations(self):
        cfg = ExperimentConfig()
        assert len(cfg.u_values) == cfg.num_locations

    def test_u_values_last_element_is_one(self):
        cfg = ExperimentConfig()
        assert cfg.u_values[-1] == pytest.approx(1.0)

    def test_u_values_geometric_ratio(self):
        cfg = ExperimentConfig()
        for i in range(cfg.num_locations - 1):
            assert cfg.u_values[i] == pytest.approx(
                cfg.u_values[i + 1] / cfg.u_increment_factor
            )

    def test_n_ut_initial_is_sum_of_middle_states(self):
        cfg = ExperimentConfig()
        expected = sum(cfg.initial_states[1:cfg.num_locations - 1])
        assert cfg.n_ut_initial == expected

    def test_n_t_initial_is_last_state(self):
        cfg = ExperimentConfig()
        assert cfg.n_t_initial == cfg.initial_states[cfg.num_locations - 1]

    def test_h_threshold_is_one_percent_of_min_cost(self):
        cfg = ExperimentConfig()
        assert cfg.h_threshold == pytest.approx(
            0.01 * min(cfg.cost_underage, cfg.cost_overage)
        )

    def test_custom_costs_affect_h_threshold(self):
        cfg = ExperimentConfig(cost_underage=100, cost_overage=200)
        assert cfg.h_threshold == pytest.approx(0.01 * 100)

    def test_custom_u_increment_changes_u_values(self):
        cfg_default = ExperimentConfig(u_increment_factor=1.1)
        cfg_custom = ExperimentConfig(u_increment_factor=1.5)
        # Higher increment → lower earlier u_values
        assert cfg_custom.u_values[0] < cfg_default.u_values[0]


class TestExperimentRunner:
    """Tests for ExperimentRunner instantiation and helper logic."""

    def test_runner_stores_config(self):
        cfg = ExperimentConfig()
        runner = ExperimentRunner(cfg)
        assert runner.config is cfg

    def test_runner_accepts_custom_config(self):
        cfg = ExperimentConfig(num_locations=5, capacity_limit=32)
        runner = ExperimentRunner(cfg)
        assert runner.config.capacity_limit == 32
