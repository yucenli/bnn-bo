from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import distributions as gdists
from laplace import Laplace
from models.model import Model
from torch import Tensor

from .utils import RegNet


class LaplacePosterior(Posterior):
    def __init__(self, posterior, output_dim):
        super().__init__()
        self.post = posterior
        self.output_dim = output_dim

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        samples = self.post.rsample(sample_shape).squeeze(-1)
        new_shape = samples.shape[:-1]
        return samples.reshape(*new_shape, -1, self.output_dim)

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        post_mean = self.post.mean.squeeze(-1)
        shape = post_mean.shape
        return post_mean.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        post_var = self.post.variance.squeeze(-1)
        shape = post_var.shape
        return post_var.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def device(self) -> torch.device:
        return self.post.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.post.dtype
    

class LaplaceBNN(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.prior_var = args["prior_var"]
        self.noise_var = args["noise_var"]
        self.likelihood = "regression"
        self.nn = RegNet(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        dtype=torch.float64,
                        device=device)
        self.bnn = None
        self.output_dim = output_dim

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        
        if len(X.shape) < 3:
            B, D = X.shape
            Q = 1
        else:
            # Transform to `(batch_shape*q, d)`
            B, Q, D = X.shape
            X = X.reshape(B*Q, D)

        K = self.num_outputs
        # Posterior predictive distribution
        # mean_y is (batch_shape*q, k); cov_y is (batch_shape*q*k, batch_shape*q*k)
        mean_y, cov_y = self._get_prediction(X)

        # Mean in `(batch_shape, q*k)`
        mean_y = mean_y.reshape(B, Q*K)

        # Cov is `(batch_shape, q*k, q*k)`
        cov_y += 1e-4*torch.eye(B*Q*K).to(X)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum('bqkbrl->bqkrl', cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q*K, Q*K)

        dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
        post_pred = GPyTorchPosterior(dist)

        if K > 1 and Q > 1:
            return LaplacePosterior(post_pred, self.output_dim)
        else:
            return post_pred

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def _get_prediction(self, test_x: torch.Tensor):
        """
        Batched Laplace prediction.

        Args:
            test_x: Tensor of size `(batch_shape, d)`.

        Returns:
            Tensor of size `(batch_shape, k)`
        """
        mean_y, cov_y = self.bnn(test_x, joint=True)

        return mean_y, cov_y

        
    def fit_and_save(self, train_x, original_train_y, save_dir):

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, original_train_y),
            batch_size=len(train_x), shuffle=True
        )
        n_epochs = 1000
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-1, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_loader))
        loss_func = torch.nn.MSELoss()

        for i in range(n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.nn(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        self.nn.eval()

        self.bnn = Laplace(
            self.nn, 
            self.likelihood,
            sigma_noise=np.sqrt(self.noise_var),
            prior_precision=(1 / self.prior_var),
            subset_of_weights='all',
            hessian_structure='kron',
            enable_backprop=True
        )
        self.bnn.fit(train_loader)
        self.bnn.optimize_prior_precision(n_steps=50)