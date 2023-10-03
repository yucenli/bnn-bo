from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .hmc_utils import run_hmc
from .model import Model
from .utils import (RegNet, make_gaussian_log_likelihood,
                    make_gaussian_log_likelihood_fixed_noise,
                    make_gaussian_log_prior)


class HMCPosterior(Posterior):
    def __init__(self, X, model, param_samples, output_dim, mean, std):
        super().__init__()
        self.model = model
        self.param_samples = param_samples
        self.output_dim = output_dim
        self.X = X
        self.preds = None
        self.y_mean = mean
        self.y_std = std

    def predict_model(self):
        self.preds = []
        for s in self.param_samples:
            torch.nn.utils.vector_to_parameters(s, self.model.parameters())
            output = self.model(self.X.to(s))[..., :self.output_dim]
            scaled_output = (output * self.y_std) + self.y_mean
            self.preds.append(scaled_output)
        self.preds = torch.stack(self.preds)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        n = len(self.param_samples)
        rand_ints = np.random.randint(n, size=sample_shape)

        if self.preds is None:
            sample = []
            for i in rand_ints:
                s = self.param_samples[i]
                torch.nn.utils.vector_to_parameters(s, self.model.parameters())
                output = self.model(self.X.to(s))[..., :self.output_dim]
                scaled_output = (output * self.y_std) + self.y_mean
                sample.append(scaled_output)
            return torch.stack(sample)
        else:
            return self.preds[rand_ints]

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        if self.preds is None:
            self.predict_model()
        return self.preds.mean(axis=0)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        if self.preds is None:
            self.predict_model()
        return self.preds.var(axis=0)

    # def _extended_shape(
    #     self, sample_shape: torch.Size = torch.Size()
    # ) -> torch.Size:
    #     r"""Returns the shape of the samples produced by the posterior with
    #     the given `sample_shape`.
    #     """
    #     return sample_shape + self.preds.shape

    @property
    def device(self) -> torch.device:
        return self.preds.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.preds.dtype


class HMC(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        # problem dimensions
        self.input_dim = input_dim
        self.problem_output_dim = output_dim

        # architecture
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.noise_var = tensor(args["noise_var"])
        self.prior_var = args["prior_var"]
        self.adapt_noise = args["adapt_noise"]
        # optional dimensions for noise estimate
        if self.adapt_noise:
            self.network_output_dim = 2 * output_dim
        else:
            self.network_output_dim = output_dim

        # HMC params
        self.n_chains = args["n_chains"]
        self.n_samples_per_chain = args["n_samples_per_chain"]
        self.n_burn_in = args["n_burn_in"]
        self.step_size = args["step_size"]
        self.path_length = args["path_length"]
        self.pretrain_steps = args["pretrain_steps"]
        self.adapt_step_size = args["adapt_step_size"]

        # standardize y values before training
        self.standardize_y = args["standardize_y"]
        self.mean = 0.0
        self.std = 1.0

        self.model = RegNet(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=self.input_dim,
                        output_dim=self.network_output_dim,
                        dtype=torch.float64,
                        device=device)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return HMCPosterior(X, self.model, self.param_samples, self.problem_output_dim, self.mean, self.std)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim

    def fit_and_save(self, train_x, original_train_y, save_dir):
        # standardize y before training
        if self.standardize_y:
            self.mean = original_train_y.mean(dim=0)
            if len(original_train_y) > 1:
                self.std = original_train_y.std(dim=0)
            else:
                self.std = 1.0
            train_y = (original_train_y - self.mean) / self.std
        else:
            train_y = original_train_y

        param_size = len(torch.nn.utils.parameters_to_vector(self.model.state_dict().values()).detach())
        self.param_samples = None
        all_params = torch.zeros([self.n_samples_per_chain * self.n_chains, param_size]).to(train_x)

        # log density functions
        weight_decay = 1.0 / self.prior_var
        log_prior_fn, log_prior_diff_fn = make_gaussian_log_prior(weight_decay, 1.)
        
        if self.adapt_noise:
            log_likelihood_fn = make_gaussian_log_likelihood(1.)
        else:
            log_likelihood_fn = make_gaussian_log_likelihood_fixed_noise(1., self.noise_var)

        def log_density_fn(model):
            log_likelihood = log_likelihood_fn(model, train_x, train_y)
            log_prior = log_prior_fn(model.parameters())
            log_density = log_likelihood + log_prior
            return log_density, log_likelihood.detach()
        
        all_likelihoods = []
        for i in range(self.n_chains):

            self.model = RegNet(dimensions=self.regnet_dims,
                                activation=self.regnet_activation,
                                input_dim=self.input_dim,
                                output_dim=self.network_output_dim,
                                dtype=train_x.dtype,
                                device=train_x.device)

            # pretrain model
            optimizer = torch.optim.Adam(self.model.parameters(), maximize=True)
            for e in range(self.pretrain_steps):
                optimizer.zero_grad()
                log_density, llh = log_density_fn(self.model)
                log_density.backward()
                if (e+1) % 1000 == 0:
                    # log_prior, log_likelihood
                    print(e + 1, log_density.item() - llh.item(), llh.item())
                optimizer.step()
            del optimizer
            self.model.zero_grad()
      
            # run HMC and save parameters
            params_hmc, llhs = run_hmc(self.n_samples_per_chain, self.model, log_density_fn, 
                                       log_prior_diff_fn, self.step_size,
                                       self.path_length, self.adapt_step_size, self.n_burn_in,
                                       True)

            all_likelihoods = all_likelihoods + llhs
            all_params[i * self.n_samples_per_chain:(i+1) * self.n_samples_per_chain] = params_hmc
            del params_hmc

        self.param_samples = all_params
        if save_dir is not None:
            plt.plot(range(len(all_likelihoods)), all_likelihoods)
            plt.savefig("%s/model_state/%d_likelihood.png" % (save_dir, len(train_x)))
            plt.clf()