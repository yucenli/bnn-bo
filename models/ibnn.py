from typing import Any, Callable, List, Optional

import botorch
import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


class IBNN_Erf(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, d, var_w, var_b, depth, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.var_w = var_w
        self.var_b = var_b
        self.depth = depth

    def k(self, l, x1, x2):
        # base case
        if l == 0:
            return self.var_b + self.var_w * (x1 * x2).sum(-1) / self.d
        else:
            K_12 = self.k(l - 1, x1, x2)
            K_11 = self.k(l - 1, x1, x1)
            K_22 = self.k(l - 1, x2, x2)
            sqrt_term = torch.sqrt((1 + 2 * K_11) * (1 + 2 * K_22))
            fraction = 2 * K_12 / sqrt_term
            epsilon = 1e-7
            theta = torch.asin(torch.clamp(fraction, min=-1 + epsilon, max=1 - epsilon))
            result = self.var_b + 2 * self.var_w / (torch.pi) * theta
            return result
        
    def forward(self, x1, x2, **params):
        d2 = x2.shape[-2]
        x1_shape = tuple(x1.shape)
        d1, dim = x1_shape[-2:]
        new_shape = x1_shape[:-2] + (d1, d2, dim)
        new_x1 = x1.unsqueeze(-2).expand(new_shape)
        new_x2 = x2.unsqueeze(-3).expand(new_shape)
        result = self.k(self.depth, new_x1, new_x2)
        return result


class IBNN_ReLU(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, d, var_w, var_b, depth, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.var_w = var_w
        self.var_b = var_b
        self.depth = depth

    def k(self, l, x1, x2):
        # base case
        if l == 0:
            return self.var_b + self.var_w * (x1 * x2).sum(-1) / self.d
        else:
            K_12 = self.k(l - 1, x1, x2)
            K_11 = self.k(l - 1, x1, x1)
            K_22 = self.k(l - 1, x2, x2)
            sqrt_term = torch.sqrt(K_11 * K_22)
            fraction = K_12 / sqrt_term
            epsilon = 1e-7
            theta = torch.acos(torch.clamp(fraction, min=-1 + epsilon, max=1 - epsilon))
            theta_term = torch.sin(theta) + (torch.pi - theta) * fraction
            result = self.var_b + self.var_w / (2 * torch.pi) * sqrt_term * theta_term
            return result
        
    def forward(self, x1, x2, **params):
        d2 = x2.shape[-2]
        x1_shape = tuple(x1.shape)
        d1, dim = x1_shape[-2:]
        new_shape = x1_shape[:-2] + (d1, d2, dim)
        new_x1 = x1.unsqueeze(-2).expand(new_shape)
        new_x2 = x2.unsqueeze(-3).expand(new_shape)
        result = self.k(self.depth, new_x1, new_x2)
        return result


class SingleTaskIBNN(Model):

    def __init__(self, model_args, input_dim, output_dim, device):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.var_b = model_args["var_b"]
        self.var_w = model_args["var_w"]
        self.depth = model_args["depth"]
        if "kernel" in model_args and model_args["kernel"] == "erf":
            self.kernel = IBNN_Erf(input_dim, self.var_w, self.var_b, self.depth)
        else:
            self.kernel = IBNN_ReLU(input_dim, self.var_w, self.var_b, self.depth)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        if self.output_dim > 1:
            raise RuntimeError(
                "SingleTaskGP does not fit tasks with multiple objectives")

        self.gp = botorch.models.SingleTaskGP(
            train_x, train_y, covar_module=self.kernel, outcome_transform=Standardize(m=1)).to(train_x)
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)


class MultiTaskIBNN(Model):

    def __init__(self, model_args, input_dim, output_dim, device):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.var_b = model_args["var_b"]
        self.var_w = model_args["var_w"]
        self.depth = model_args["depth"]
        if model_args["kernel"] == "erf":
            self.kernel = IBNN_Erf(input_dim, self.var_w, self.var_b, self.depth)
        else:
            self.kernel = IBNN_ReLU(input_dim, self.var_w, self.var_b, self.depth)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        models = []
        for d in range(self.output_dim):
            models.append(
                botorch.models.SingleTaskGP(
                    train_x,
                    train_y[:, d].unsqueeze(-1),
                    covar_module=self.kernel,
                    outcome_transform=Standardize(m=1)).to(train_x))

        self.gp = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)
