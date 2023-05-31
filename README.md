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

