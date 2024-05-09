import math
import time
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .model import Model
from .utils import (RegNet, make_gaussian_log_likelihood,
                    make_gaussian_log_likelihood_fixed_noise,
                    make_gaussian_log_prior)


class DropoutPosterior(Posterior):
    def __init__(self, X, model, output_dim, noise_var,n_samples,mean,std):
        super().__init__()
        self.model = model
        self.output_dim = output_dim
        self.noise_var = noise_var
        self.n_samples = n_samples
        self.y_mean = mean
        self.y_std = std

        self.predict_model(X)

    def predict_model(self, X):
        self.model.train()
        preds = [(self.model(X)*self.y_std+self.y_mean) for _ in range(self.n_samples)]
        self.pred_mean = torch.stack(preds)


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


class Dropout(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        self.n_models = args["n_models"]
        self.train_prop = args["train_prop"]
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.train_steps = args["train_steps"]
        self.prior_var = 1.0 / args["prior_var"]
        self.noise_var = torch.tensor(args["noise_var"])
        self.n_samples = args["n_samples"]

        self.input_dim = input_dim
        self.problem_output_dim = output_dim
        self.network_output_dim = output_dim

        self.dropout_prob = args["dropout_prob"]  # Add dropout probability
        self.mean = 0
        self.std = 1
        self.standardize_y = args["standardize_y"]

        self.model = RegNetWithDropout(
            dimensions=self.regnet_dims,
            activation=self.regnet_activation,
            input_dim=self.input_dim,
            output_dim=self.network_output_dim,
            dropout_prob=self.dropout_prob,  # Pass dropout probability to the model
            dtype=torch.float64,
            device=device
        )
        #self.param_samples = None

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return DropoutPosterior(X, self.model, self.problem_output_dim, self.noise_var,self.n_samples,self.mean,self.std)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim

    def fit_and_save(self, train_x, original_train_y, save_dir):
        if self.standardize_y:
            self.mean = original_train_y.mean(dim=0)
            if len(original_train_y) > 1:
                self.std = original_train_y.std(dim=0)
            else:
                self.std = 1.0
            train_y = (original_train_y - self.mean) / self.std
        else:
            train_y = original_train_y
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()  # Assuming a regression problem

        self.model.train()
        for _ in range(self.train_steps):  # Assuming 1000 training steps
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()



class RegNetWithDropout(RegNet):
    def __init__(self, dimensions, activation, input_dim, output_dim, dropout_prob, **kwargs):
        super().__init__(dimensions, activation, input_dim, output_dim, **kwargs)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout(x)  # Apply dropout
        return x