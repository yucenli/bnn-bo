from typing import Any, Callable, List, Optional
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .hmc_utils import run_hmc
from .model import Model
from .utils import (RegNet, BNN)
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
from torch import Tensor, tensor
import torch
import numpy as np
from botorch.posteriors import Posterior
from typing import Any, Callable, List, Optional

class MYNUTSPosterior(Posterior):
    def __init__(self, X, model, samples,param_samples, output_dim, mean,inter_model,std):
        super().__init__()
        self.model = model
        self.samples = samples
        self.param_samples = param_samples
        self.output_dim = output_dim
        self.X = X
        self.preds = None
        self.y_mean = mean
        self.y_std = std
        self.inter_model = inter_model



        
    def predict_model(self):
        self.preds = []
        for s in self.param_samples:
            torch.nn.utils.vector_to_parameters(s, self.inter_model.parameters())
            output = self.inter_model(self.X.to(s))[..., :self.output_dim]
            output = (output * self.y_std) + self.y_mean
            self.preds.append(output)
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
                torch.nn.utils.vector_to_parameters(s, self.inter_model.parameters())
                output = self.inter_model(self.X.to(s))[..., :self.output_dim]
                output = (output * self.y_std) + self.y_mean
                sample.append(output)
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


class MYNUTS(Model):
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
        self.adapt_step_size = args["adapt_step_size"]
        self.prior_var = args["prior_var"]
        # standardize y values before training
        self.standardize_y = args["standardize_y"]
        self.mean = 0.0
        self.std = 1.0

        self.model = BNN(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=self.input_dim,
                        output_dim=self.network_output_dim,
                        prior_var=self.prior_var,
                        dtype=torch.float64,
                        device=device)
        self.inter_model = RegNet(dimensions=self.regnet_dims,
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
        return MYNUTSPosterior(X, self.model,self.samples, self.param_samples, self.problem_output_dim, self.mean,self.inter_model,self.std)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim

    def fit_and_save(self, train_x, original_train_y, save_dir):
        #train_x = train_x.squeeze()
        #train_y = train_y.squeeze()

        if self.standardize_y:
            self.mean = original_train_y.mean(dim=0)
            if len(original_train_y) > 1:
                self.std = original_train_y.std(dim=0)
            else:
                self.std = 1.0
            train_y = (original_train_y - self.mean) / self.std
        else:
            train_y = original_train_y


        nuts_kernel = NUTS(self.model, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=self.n_samples_per_chain, num_chains=self.n_chains, warmup_steps=self.n_burn_in)
        mcmc.run(train_x, train_y)

        self.samples= mcmc.get_samples()
        self.concat_samples = copy.deepcopy(self.samples)
        for key in self.concat_samples:
            self.concat_samples[key] = self.concat_samples[key].view(self.n_samples_per_chain, -1)

        # Concatenate the tensors
        self.param_samples = torch.cat(list(self.concat_samples.values()), dim=1)

                


        
        
