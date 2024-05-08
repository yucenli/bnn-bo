from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from models.model import Model
from torch import Tensor

from pybnn.bohamiann import Bohamiann
from pybnn.util.layers import AppendLayer

from .utils import RegNet, get_best_hyperparameters


class SGHMCPosterior(Posterior):
    def __init__(self, X, model):
        super().__init__()
        _, _, samples = model.predict(X, return_individual_predictions=True)
        self.preds = samples

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        n = len(self.preds)
        rand_ints = np.random.randint(n, size=sample_shape)
        return self.preds[rand_ints]

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        return self.preds.mean(axis=0)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        return self.preds.var(axis=0)

    @property
    def device(self) -> torch.device:
        return self.preds.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.preds.dtype



class SGHMCModel(Model):

    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.device = device
        self.batch_size = args["batch_size"] if "batch_size" in args else 5

        # architecture
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.prior_var = args["prior_var"] if "prior_var" in args else 1.0
        self.noise_var = args["noise_var"] if "noise_var" in args else torch.tensor(1.0)

        self.iterative = args["iterative"] if "iterative" in args else True

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return SGHMCPosterior(X, self.model)

    @property
    def num_outputs(self) -> int:
        return self.output_dim
    
    def fit_sghmc_model(self, train_x, train_y, prior_var, noise_var):
        def get_network(input_dimensionality, output_dim, device) -> torch.nn.Module:
            regnet = RegNet(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=input_dimensionality,
                        output_dim=output_dim,
                        dtype=torch.float64,
                        device=device)
            last_layer = AppendLayer(noise=noise_var, device=device)
            return torch.nn.Sequential(regnet, last_layer)
        
        model = Bohamiann(get_network, prior_var=prior_var, print_every_n_steps=1000, device=self.device)
        model.train(train_x, train_y, num_steps=2000, num_burn_in_steps=1000, keep_every=5, lr=1e-2, batch_size=self.batch_size, verbose=False)
        return model

    def get_likelihood(self, train_x, train_y, prior_var, noise_var):
        n = len(train_x)
        n_train = int(0.8 * n)
        train_x, val_x = train_x[:n_train], train_x[n_train:]
        train_y, val_y = train_y[:n_train], train_y[n_train:]

        model = self.fit_sghmc_model(train_x, train_y, prior_var, noise_var)
        posterior = SGHMCPosterior(val_x, model)
        
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

        self.model = self.fit_sghmc_model(train_x, train_y, self.prior_var, self.noise_var)
