from typing import Optional

import numpy as np
import torch
from botorch.test_functions.base import BaseTestProblem

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(BaseTestProblem):
    """
	Pest Control Problem.

	"""

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False
    ) -> None:
        self.dim = PESTCONTROL_N_STAGES
        self._bounds = np.repeat([[0], [5 - 1e-6]], PESTCONTROL_N_STAGES, axis=1).T
        self.num_objectives = 1

        super().__init__(
            noise_std=noise_std,
            negate=negate,
        )

        self.categorical_dims = np.arange(self.dim)

    def evaluate_true(self, X):
        res = torch.stack([self._compute(x) for x in X]).to(X)
        # Add a small amount of noise to prevent training instabilities
        res += 1e-6 * torch.randn_like(res)
        return res

    def _compute(self, x):
        evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy(), seed=1)
        return torch.tensor(evaluation)