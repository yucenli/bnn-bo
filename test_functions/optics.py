from typing import Optional

import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor

try:
    from gym_interf import InterfEnv
except:
    pass

class Optics(BaseTestProblem):
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        aggregate: bool = True
    ) -> None:

        self.env = None
        self.dim = 4
        self._bounds = [
            [-1.0, 1.0], 
            [-1.0, 1.0], 
            [-1.0, 1.0], 
            [-1.0, 1.0]
        ]
        self.num_objectives = 1
        super().__init__(
                negate=negate, noise_std=noise_std)


    def metric(self, image):
        xx, yy = torch.meshgrid(
            torch.arange(64, dtype=image.dtype, device=image.device) / 64,
            torch.arange(64, dtype=image.dtype, device=image.device) / 64,
        )
        intens = (-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2).exp() * image
        ivec = intens.sum((-1, -2))
        smax = torch.logsumexp(ivec, -1)
        smin = -torch.logsumexp(-ivec, -1)
        v = (smax - smin) / (smax + smin)
        return v #v.log() - (1 - v).log()

    def cfun(self, x):
        self.env = InterfEnv()
        self.env.seed(1)
        self.env.reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))

        action = x[:4].cpu().detach().numpy()
        state = self.env.step(action)
        return torch.tensor(state[0])

    def c_batched(self, X):
        return torch.stack([self.cfun(x) for x in X]).to(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        images = self.c_batched(X)
        return torch.stack([self.metric(image) for image in images]).to(X)
    
    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        return super().forward(X, False)