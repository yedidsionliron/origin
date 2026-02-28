import sys
import os

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'Newsvendor-last-mile', 'Experiments-code'),
)
from dp_lookup import DPXGBoostApproximator


def test_fit_and_predict():
    # simple environment from demonstration
    def reward(s, a):
        return float(- (s - a) ** 2)

    def transition(s, a):
        return np.array([s * 0.5 + a])

    samples = np.linspace(0, 1, 5).reshape(-1, 1)
    approximator = DPXGBoostApproximator(
        horizon=2,
        action_space=[0, 1],
        reward_fn=reward,
        transition_fn=transition,
    )
    approximator.fit(samples)

    # after fit we should have models for each timestep
    assert len(approximator.models) == 2

    preds0 = approximator.predict(samples, 0)
    preds1 = approximator.predict(samples, 1)
    assert preds0.shape == (5,)
    assert preds1.shape == (5,)
