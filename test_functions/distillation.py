from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .meta_module import *


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
     

class LeNetSmall(MetaModule):

    def __init__(self, n_classes):
        super(LeNetSmall, self).__init__()
        
        self.conv1 = MetaConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = MetaConv2d(in_channels=16, out_channels=12, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.classifier = MetaLinear(in_features=432, out_features=n_classes)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def vector_to_parameters(params, model):
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


class KnowledgeDistillation(SyntheticTestFunction):

    def __init__(
        self,
        noise_std = None,
        negate: bool = False
    ) -> None:
        

        student = LeNetSmall(10)
        param_size = len(torch.nn.utils.parameters_to_vector(student.state_dict().values()))
        print(param_size)

        self.dim = param_size
        self.num_objectives = 1
        self._bounds = np.repeat([[-1, 1]], self.dim, axis=0)
        super().__init__(negate=negate, noise_std=noise_std)

        self.student = student#.to("cuda")

        self.teacher = LeNet5(10).to("cuda")
        self.teacher.load_state_dict(torch.load("lenet_mnist.pt"))

        train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])

        train_dataset = datasets.MNIST(root='/datasets', 
                                    train=True, 
                                    transform=train_transforms,
                                    download=False)

        # define the data loaders
        self.train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=1000, 
                                shuffle=True)

    def f(self, params):
        vector_to_parameters(params, self.student)
        loss = nn.KLDivLoss(reduction="sum")
        for data, target in self.train_loader:
            student_probs = self.student(data.to("cuda"))[1] + 1e-10
            teacher_probs = self.teacher(data.to("cuda"))[1] + 1e-10

            return loss(teacher_probs, student_probs).detach()
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.stack([self.f(x) for x in X]).to(X)
        return result