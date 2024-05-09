"""Likelihoods, priors, and RegNet."""

import math
import torch
from torch.nn import functional as F
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
from torch import Tensor, tensor
import torch
import numpy as np
from .model import Model
from botorch.posteriors import Posterior

def make_gaussian_log_prior(weight_decay, temperature):
  """Returns the Gaussian log-density and delta given weight decay."""

  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    params = list(params)
    n_params = sum([p.numel() for p in params])
    param_normsq = sum([(p**2).sum() for p in params])
    log_prob = -(0.5 * param_normsq * weight_decay +
                 0.5 * n_params * np.log(weight_decay / (2 * math.pi)))
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior log-density."""
    diff = sum([(p1 ** 2 - p2 ** 2).sum() for p1, p2 in
                zip(params1, params2)])
    return -0.5 * weight_decay * diff / temperature

  return log_prior, log_prior_diff


def preprocess_network_outputs_gaussian(predictions):
  """Apply softplus to std output.

  Returns predictive mean and standard deviation.
  """
  num_dims = int(predictions.shape[-1] / 2)
  predictions_mean, predictions_std = torch.split(predictions, [num_dims, num_dims], dim=-1)
  predictions_std = F.softplus(predictions_std)
  return predictions_mean, predictions_std


def make_gaussian_log_likelihood(temperature):
  def gaussian_log_likelihood(model, x, y, batch_size=None):
    """Computes the negative log-likelihood.

    The outputs of the network should be two-dimensional.
    The first output is treated as predictive mean. The second output is treated
    as inverse-softplus of the predictive standard deviation.
    """
    if batch_size is not None:
      indices = torch.randint(len(x), (batch_size,))
      x = x[indices]
      y = y[indices]
    predictions = model(x)
    predictions_mean, predictions_std = (
        preprocess_network_outputs_gaussian(predictions))
    tempered_std = torch.clamp_min(predictions_std * np.sqrt(temperature), 1e-10)
    se = (predictions_mean - y) ** 2
    log_likelihood = (-0.5 * se / tempered_std ** 2
                      - 0.5 * torch.log(tempered_std ** 2 * 2 * math.pi))
    log_likelihood = log_likelihood.sum()

    return log_likelihood

  return gaussian_log_likelihood


def make_gaussian_log_likelihood_fixed_noise(temperature, noise):
  def gaussian_log_likelihood(model, x, y, batch_size=None):
    """Computes the negative log-likelihood.

    The outputs of the network should be two-dimensional.
    The first output is treated as predictive mean. The second output is treated
    as inverse-softplus of the predictive standard deviation.
    """
    if batch_size is not None:
      indices = torch.randint(len(x), (batch_size,))
      x = x[indices]
      y = y[indices]
    predictions_mean = model(x)
    predictions_std = noise
    # predictions_mean, predictions_std = (
    #     preprocess_network_outputs_gaussian(predictions))
    tempered_std = predictions_std * np.sqrt(temperature)
    se = (predictions_mean - y) ** 2
    log_likelihood = (-0.5 * se / tempered_std ** 2
                      - 0.5 * torch.log(tempered_std ** 2 * 2 * math.pi))
    log_likelihood = log_likelihood.sum()

    return log_likelihood

  return gaussian_log_likelihood


class RegNet(torch.nn.Sequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1,
                        dtype=torch.float64, device="cpu"):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU())
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)


class BNN(PyroModule):
    def __init__(self,dimensions, activation, input_dim=1, output_dim=1,prior_var=5,
                        dtype=torch.float64, device="cpu"):
        super().__init__()
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.input_dim = input_dim
        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [input_dim] + dimensions + [output_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_var * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_var).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
            # Ensure x is at least 2D tensor
            if x.dim() == 1:
                x = x.unsqueeze(1)  # Add singleton dimension if input is 1D

            # Forward pass through the network
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            mu = self.layers[-1](x)

            # Infer the response noise
            sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))

            # Sample observations
            with pyro.plate("data", size=x.shape[0], subsample=y):
                obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
            return mu