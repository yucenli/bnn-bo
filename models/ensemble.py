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
                    make_gaussian_log_prior,
                    get_best_hyperparameters)


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
        self.prior_var = args["prior_var"] if "prior_var" in args else 1.0
        self.noise_var = args["noise_var"] if "noise_var" in args else torch.tensor(1.0)

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

        self.iterative = args["iterative"] if "iterative" in args else True

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
    
    def fit_ensemble(self, train_x, train_y, prior_var, noise_var):
        all_params = tensor([]).to(train_x)
        for i in range(self.n_models):
            n_train_samples = math.ceil(self.train_prop * len(train_x))
            indices = torch.randperm(len(train_x))[:n_train_samples]
            model_train_x = train_x[indices]
            model_train_y = train_y[indices]
        
            log_prior_fn, log_prior_diff_fn = make_gaussian_log_prior(1.0 / prior_var, 1.)
            log_likelihood_fn = make_gaussian_log_likelihood_fixed_noise(1., noise_var)

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
        return all_params
    
    def get_likelihood(self, train_x, train_y, prior_var, noise_var):
        n = len(train_x)
        n_train = int(0.8 * n)
        train_x, val_x = train_x[:n_train], train_x[n_train:]
        train_y, val_y = train_y[:n_train], train_y[n_train:]

        params = self.fit_ensemble(train_x, train_y, prior_var, noise_var)
        posterior = NNPosterior(val_x, self.model, params, self.problem_output_dim, noise_var)
    
        predictions_mean = posterior.mean
        # std combines epistemic and aleatoric uncertainty
        predictions_std = torch.sqrt(posterior.variance + self.noise_var)
        # get log likelihood
        likelihood = torch.distributions.Normal(predictions_mean, predictions_std).log_prob(val_y).sum()
        return likelihood

    def fit_and_save(self, train_x, train_y, save_dir):
        if self.iterative:
            llh_fn = self.get_likelihood
            self.prior_var, self.noise_var = get_best_hyperparameters(train_x, train_y, llh_fn)
            
        all_params = self.fit_ensemble(train_x, train_y, self.prior_var, self.noise_var)
        self.param_samples = all_params