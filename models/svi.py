from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .model import Model
from .utils import (RegNet, BNN)

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class SVIPosterior(Posterior):
    def __init__(self, X, model, output_dim,guide,inter_model,mean,std):
        super().__init__()
        self.model = model
        self.output_dim = output_dim
        self.X = X
        self.preds = None
        self.guide = guide
        self.inter_model = inter_model
        self.y_mean = mean
        self.y_std = std

    def predict_model(self):
        self.preds = []
        for _ in range(500):
            a = self.guide()
            for key in a:
                a[key] = a[key].view(1, -1)
            s = torch.cat(list(a.values()), dim=1)[0]
            torch.nn.utils.vector_to_parameters(s, self.inter_model.parameters())
            output = self.inter_model(self.X.to(s))[..., :self.output_dim]
            output = (output * self.y_std) + self.y_mean
            self.preds.append(output)
        self.preds = torch.stack(self.preds)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        rand_ints = np.random.randint(50, size=sample_shape)

        if self.preds is None:
            sample = []

            for i in range(len(rand_ints)):
                a = self.guide()
                for key in a:
                    a[key] = a[key].view(1, -1)
                s = torch.cat(list(a.values()), dim=1)[0]
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


class MySVI(Model):
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
        self.num_epochs = args["num_epochs"]
        # optional dimensions for noise estimate
        self.standardize_y = args["standardize_y"]
        self.mean = 0.0
        self.std = 1.0
        if self.adapt_noise:
            self.network_output_dim = 2 * output_dim
        else:
            self.network_output_dim = output_dim



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
        return SVIPosterior(X, self.model, self.problem_output_dim, self.guide,self.inter_model,self.mean,self.std)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim
    def fit_and_save(self, train_x, original_train_y,save_dir):
        if self.standardize_y:
            self.mean = original_train_y.mean(dim=0)
            if len(original_train_y) > 1:
                self.std = original_train_y.std(dim=0)
            else:
                self.std = 1.0
            train_y = (original_train_y - self.mean) / self.std
        else:
            train_y = original_train_y

        #train_x = train_x.squeeze()
        #train_y = train_y.squeeze()
        self.guide = AutoDiagonalNormal(self.model)
        optimizer = pyro.optim.Adam({"lr": 0.01})

        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()

        progress_bar = trange(self.num_epochs)

        for epoch in progress_bar:
            loss = svi.step(train_x, train_y)
            progress_bar.set_postfix(loss=f"{loss / train_x.shape[0]:.3f}")




