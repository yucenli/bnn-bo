import math
import time
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .model import Model
from .utils import (RegNet, make_gaussian_log_likelihood,
                    make_gaussian_log_likelihood_fixed_noise,
                    make_gaussian_log_prior)


class NNPosterior(Posterior):
    def __init__(self, X, model, param_samples, output_dim, noise_var):
        super().__init__()
        self.model = model
        self.param_samples = param_samples
        self.output_dim = output_dim
        self.noise_var = noise_var

        self.predict_model(X)

    def predict_model(self, X):
        self.pred_mean = []
        for s in self.param_samples:
            torch.nn.utils.vector_to_parameters(s, self.model.parameters())
            output = self.model(X.to(s))
            self.pred_mean.append(output)
        self.pred_mean = torch.stack(self.pred_mean)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        n = len(self.pred_mean)
        i = np.random.randint(n, size=sample_shape)
        mean = self.pred_mean[i]
        sample = torch.randn_like(mean) * self.noise_var.sqrt() + mean
        return sample

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        return self.pred_mean.mean(axis=0)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        return self.pred_mean.var(axis=0) + self.noise_var

    @property
    def device(self) -> torch.device:
        return self.pred_mean.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.pred_mean.dtype


class Ensemble(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        self.n_models = args["n_models"]
        self.train_prop = args["train_prop"]
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.train_steps = args["train_steps"]
        self.prior_var = 1.0 / args["prior_var"]
        self.noise_var = torch.tensor(args["noise_var"])

        self.input_dim = input_dim
        self.problem_output_dim = output_dim
        self.network_output_dim = output_dim

        self.model = RegNet(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=self.input_dim,
                        output_dim=self.network_output_dim,
                        dtype=torch.float64,
                        device=device)
        self.param_samples = None

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return NNPosterior(X, self.model, self.param_samples, self.problem_output_dim, self.noise_var)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim

    def fit_and_save(self, train_x, train_y, save_dir):
        all_params = tensor([]).to(train_x)
        for i in range(self.n_models):
            print("training", i)
            n_train_samples = math.ceil(self.train_prop * len(train_x))
            indices = torch.randperm(len(train_x))[:n_train_samples]
            model_train_x = train_x[indices]
            model_train_y = train_y[indices]
        
            log_prior_fn, log_prior_diff_fn = make_gaussian_log_prior(1.0 / self.prior_var, 1.)
            log_likelihood_fn = make_gaussian_log_likelihood_fixed_noise(1., self.noise_var)

            def log_density_fn(model):
                log_likelihood = log_likelihood_fn(model, model_train_x, model_train_y)
                log_prior = log_prior_fn(model.parameters())
                log_density = log_likelihood + log_prior
                return log_density, log_likelihood.detach()

            # MAP estimate
            net = RegNet(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=self.input_dim,
                        output_dim=self.network_output_dim,
                        dtype=train_x.dtype,
                        device=train_x.device)
            optimizer = torch.optim.Adam(net.parameters(), maximize=True)
            for e in range(self.train_steps):
                optimizer.zero_grad()
                log_density, llh = log_density_fn(net)
                log_density.backward()
                optimizer.step()

            params = torch.nn.utils.parameters_to_vector(net.state_dict().values()).to(train_x).unsqueeze(0)
            all_params = torch.cat((all_params, params))
            del params

        self.param_samples = all_params