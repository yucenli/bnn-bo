from typing import Optional

import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor

try:
    from pde import PDE, FieldCollection, ScalarField, UnitGrid
except:
    pass

class PDEVar(BaseTestProblem):
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        aggregate: bool = False
    ) -> None:

        self.dim = 4
        self._bounds = [
            [0.1, 5.0], 
            [0.1, 5.0], 
            [0.01, 5.0], 
            [0.01, 5.0],    
        ]
        self.num_objectives = 1
        super().__init__(
                negate=negate, noise_std=noise_std)

    def Simulator(self, tensor):
            a = tensor[0].item()
            b = tensor[1].item()
            d0 = tensor[2].item()
            d1 = tensor[3].item()

            eq = PDE(
                {
                    "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                    "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
                }
            )

            # initialize state
            grid = UnitGrid([64, 64])
            u = ScalarField(grid, a, label="Field $u$")
            v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
            state = FieldCollection([u, v])

            sol = eq.solve(state, t_range=20, dt=1e-3)
            sol_tensor = torch.stack(
                (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
            )
            sol_tensor[~torch.isfinite(sol_tensor)] = 1e5 * torch.randn_like(
                sol_tensor[~torch.isfinite(sol_tensor)]
            )
            return sol_tensor

    
    def evaluate_true(self, X: Tensor) -> Tensor:
        # Evaluate the simulator on each of the inputs in batch
        sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

        # Extract the size of the grid in the simulator
        sz = sims.shape[-1]

        # Create a weighting array where the edges have a weight of 1/10
        # and the middle has a weight of 1
        weighting = (
            torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
        )
        weighting[:, [0, 1, -2, -1], :] = 1.0
        weighting[:, :, [0, 1, -2, -1]] = 1.0

        # Apply the weighting to the simulator outputs
        sims = sims * weighting

        # Calculate the variance of the weighted simulator outputs
        return sims.var(dim=(-1, -2, -3))