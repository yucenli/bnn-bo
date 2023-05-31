import os
from functools import reduce

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor

from .meta_module import *


class MetaRegNet(MetaSequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1, 
                        dtype=torch.float64, device="cpu"):
        super(MetaRegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, MetaLinear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU())
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)


class BnnDraw(MultiObjectiveTestProblem):

    def __init__(
        self,
        input_dim,
        output_dim,
        seed,
        noise_std = None,
        negate: bool = False
    ) -> None:
        self.dim = input_dim
        self.num_objectives = output_dim
        self._bounds = np.repeat([[0, 1]], input_dim, axis=0)
        self._ref_point = np.ones(output_dim) * -2
        super().__init__(negate=negate, noise_std=noise_std)

        dimensions = [256, 256]
        activation = "tanh"
        self.model = MetaRegNet(dimensions=dimensions,
                                activation=activation,
                                input_dim=input_dim,
                                output_dim=output_dim,
                                dtype=torch.float64)
        path = "%s/bnn_params/%d_%d_%d_%s" % (os.getcwd(), input_dim, output_dim, seed, activation)
        for d in dimensions:
            path = path + "_%d" % d
        path = path + ".pt"
        if True and os.path.exists(path):
            self.params = torch.load(path)
            self.vector_to_parameters(self.params, self.model)
        else:
            os.makedirs("%s/bnn_params" % os.getcwd())
            param_size = len(torch.nn.utils.parameters_to_vector(self.model.state_dict().values()).detach())

            self.params = torch.distributions.Normal(torch.zeros(param_size), 1.0).sample()
            self.vector_to_parameters(self.params, self.model)
            torch.save(self.params, path)

    def vector_to_parameters(self, params, model):
        pointer = 0

        def get_module_by_name(module,
                            access_string: str):
            """Retrieve a module nested in another by its access string.

            Works even when there is a Sequential in the module.
            """
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)
        
        for name, param in model.named_parameters():
            old_params = get_module_by_name(model, name)
            old_params = old_params + 1
            # The length of the parameter
            num_param = param.numel()

            new_params = params[pointer:pointer + num_param].view_as(param)
            exec("model." + name + " = new_params")
            
            # Increment the pointer
            pointer += num_param


    def f(self, x):
        return self.model(x)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.model(X.cpu()).squeeze(-1).to(X)


class PolyDrawIndependent(SyntheticTestFunction):

    def __init__(
        self,
        input_dim,
        seed,
        noise_std = None,
        negate: bool = False
    ) -> None:
        self.dim = input_dim
        self.num_objectives = 1
        self._bounds = np.repeat([[0, 1]], input_dim, axis=0)
        super().__init__(negate=negate, noise_std=noise_std)

        path = "%s/bnn_params/poly_%d_%d.pt" % (os.getcwd(), input_dim, seed)
        if True and os.path.exists(path):
            self.coef = torch.load(path)
        else:
            self.coef = torch.rand((input_dim,)) * 2 - 1
            torch.save(self.coef, path)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (torch.pow(X, self.dim) * self.coef.to(X)).sum(-1)


class PolyDraw(SyntheticTestFunction):

    def __init__(
        self,
        input_dim,
        seed,
        noise_std = None,
        negate: bool = False
    ) -> None:
        self.dim = input_dim
        if self.dim % 4 != 0:
            raise RuntimeError("Dimension must be divisible by 4")
        self.num_objectives = 1
        self._bounds = np.repeat([[0, 1]], input_dim, axis=0)
        super().__init__(negate=negate, noise_std=noise_std)

        path = "%s/bnn_params/polynew_%d_%d.pt" % (os.getcwd(), input_dim, seed)
        if True and os.path.exists(path):
            self.c = torch.load(path)
        else:
            self.c = torch.randn((input_dim,))
            torch.save(self.c, path)

    def f(self, x):
        return torch.pow(x, torch.range(len(x)))

    def evaluate_true(self, X: Tensor) -> Tensor:
        # (x_1 - c_1) * (x_2 - c_2) * (x_3 - c_3) * (x_4 - c_4)
        # + (x_5 - c_5) * (x_6 - c_6) * (x_7 - c_7) * (x_8 - c_8)
        # + ...
        X = X - self.c.to(X)
        X_reshape = X.reshape(*X.shape[:-1], 4, int(self.dim / 4))
        X_mult = X_reshape.prod(-2)
        return X_mult.sum(-1)