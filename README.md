# A Study of Bayesian Neural Network Surrogates for Bayesian Optimization

## Installation
Create a new conda environment:
````
conda env create -f environment.yml
````

Install the project:
````
pip install -e .
````

## Running experiments
Each experiment requires a config json, and there many examples of config files in `config`. 

To use the config file `config/<name>.json`, run the following command from the root folder
````
python main.py --config <name>
````

You can also include the `--bg` flag if you would like to redirect stderr and stdout to a different file and save the outputs.
````
python main.py --config <name> --bg
````

## Code Organization
The Bayesian optimization loop is in `main.py`

`models`: the model code for each of the surrogate models we consider.

`test_functions`: objective functions for benchmark problems


## Adding a New Test Function

### Defining the function

Our library supports any test function which extends the `BaseTestProblem` class defined in BoTorch ([documentation](https://botorch.org/api/test_functions.html#botorch.test_functions.base.BaseTestProblem)). This class requires an implementation of the `evaluate_true` method, which takes in X values and returns the value of the objective function at those values.

For example, in order to specify the objective function $y = x^2$, we can define the following class:
```
class Toy(BaseTestProblem):
    dim = 1

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.pow(X, 2)
```

Many of the test function we use in the library are defined in the `test_functions` folder, or directly imported from BoTorch.

### Adding logic to main.py

To add a new test function, we need to modify `get_test_function` in `main.py`. This function will parse the string specified in the config file by "test_function" and use it to initialize the test function. 

## Adding a New Surrogate Model

### Defining the model

The core logic for each model lies in the `model` folder. To add a new surrogate model implementation, add a new class that extends the `Model` class (`model.model`). This class extends the botorch.model.model.Model class ([documentation](https://botorch.org/api/models.html#model-apis)).

The two most important functions to implement are `posterior`, which computes the posterior at the specified points ([BoTorch documentation](https://botorch.org/api/models.html#botorch.models.model.Model.posterior)), and `fit_and_save`, which fits the model to the queried points.

### Adding logic to main.py

To run the new surrogate model and compare it with other models, we will need to modify the `initialize_model` function in `main.py` in order to parse the config file and initialize the model appropriately.