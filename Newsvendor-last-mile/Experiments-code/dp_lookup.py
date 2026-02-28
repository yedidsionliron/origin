"""DP lookup-table approximation using gradient-boosted tree regressors.

This module implements a finite-horizon dynamic programming approximator
where value functions are represented by gradient-boosted tree regressors.
It is structured according to the project's global coding guidelines:

- object-oriented design (single class `DPXGBoostApproximator`)
- vectorized operations via NumPy / pandas
- self-contained in one file
- minimal external dependencies (scikit-learn, numpy, pandas)

The main class can be used to sequentially fit models $f_t$ that approximate
$G_t(s)$ at each time step.  A small runnable demonstration and a basic
entry-point are provided at the bottom of the file.
"""

from __future__ import annotations
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class DPXGBoostApproximator:
    """Finite-horizon DP approximator using XGBoost regressors.

    Attributes:
        horizon (int): number of time steps.
        models (List[XGBRegressor]): fitted regressors for each time step.
        discount (float): discount factor gamma.
        action_space (List[Any]): discrete set of possible actions.
        reward_fn (Callable): R(s, a) immediate reward.
        transition_fn (Callable): function mapping (s, a) -> next state.
    """

    def __init__(
        self,
        horizon: int,
        action_space: List[Any],
        reward_fn: Callable[[np.ndarray, Any], float],
        transition_fn: Callable[[np.ndarray, Any], np.ndarray],
        discount: float = 0.99,
        model_params: Optional[dict] = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        self.horizon = horizon
        self.discount = discount
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.transition_fn = transition_fn
        self.models: List[GradientBoostingRegressor] = []
        self.model_params = model_params or {}

    def fit(self, state_samples: np.ndarray) -> None:
        """Fit a separate regressor at each time step.

        ``state_samples`` is an (N, d) array of representative states over
        which we will approximate the value function.  Models are trained
        backwards from t = horizon-1 to 0.
        """
        N = state_samples.shape[0]
        # initialize next-step values to zero
        next_values = np.zeros(N)

        # backward induction
        for t in reversed(range(self.horizon)):
            # compute Bellman targets for each state sample and action
            targets = np.zeros(N)
            for i, s in enumerate(state_samples):
                # evaluate best action
                best = -np.inf
                for a in self.action_space:
                    r = self.reward_fn(s, a)
                    s_next = self.transition_fn(s, a)
                    # for simplicity assume next_values corresponds to s_next index
                    # in practice we would interpolate or evaluate model
                    val = r + self.discount * self._eval_next(s_next, next_values, state_samples)
                    if val > best:
                        best = val
                targets[i] = best

            # fit model for step t
            model = GradientBoostingRegressor(**self.model_params)
            model.fit(state_samples, targets)
            self.models.insert(0, model)  # prepend so index corresponds to t
            next_values = targets.copy()

    def _eval_next(self, s_next: np.ndarray, next_values: np.ndarray, states: np.ndarray) -> float:
        """Helper: approximate value of a next state using nearest neighbor.

        This naive implementation simply takes the value of the closest sample.
        """
        # compute distances
        dists = np.linalg.norm(states - s_next, axis=1)
        idx = int(np.argmin(dists))
        return next_values[idx]

    def predict(self, states: np.ndarray, time: int) -> np.ndarray:
        """Predict value for given states at a particular time step."""
        if time < 0 or time >= len(self.models):
            raise ValueError("Time index out of range")
        return self.models[time].predict(states)

    def __repr__(self) -> str:
        return f"DPXGBoostApproximator(horizon={self.horizon}, models={len(self.models)})"


# demonstration / simple smoke test
if __name__ == "__main__":
    # toy environment: state is scalar, two actions {0,1}
    def reward(s, a):
        return float(- (s - a) ** 2)

    def transition(s, a):
        return np.array([s * 0.5 + a])

    # sample states
    samples = np.linspace(0, 1, 10).reshape(-1, 1)
    approximator = DPXGBoostApproximator(
        horizon=3,
        action_space=[0, 1],
        reward_fn=reward,
        transition_fn=transition,
    )
    approximator.fit(samples)
    print(approximator)
    print("prediction at t=0", approximator.predict(samples, 0))
