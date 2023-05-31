import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import \
    qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.test_functions import *
from botorch.utils.multi_objective.box_decompositions.dominated import \
    DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import \
    FastNondominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from models import *
from test_functions import (BnnDraw, KnowledgeDistillation, LunarLanderProblem,
                            OilSorbent, Optics, PDEVar, PestControl, PolyDraw,
                            cco)


def round(test_function_name, x):
    if test_function_name == "oil":
        x[..., 2:] = torch.floor(x[..., 2:])
    elif test_function_name == "cco":
        x[..., 15:] = torch.floor(x[..., 15:])
    elif test_function_name == "pest":
        x = torch.floor(x)
    return x


def bayes_opt(model, test_function, args, init_x, init_y, model_save_dir, device, model_name, test_function_name):
    q = int(args["batch_size"])
    output_dim = init_y.shape[-1]
    bounds = test_function.bounds.to(init_x)

    standard_bounds = torch.zeros(2, test_function.dim).to(init_x)
    standard_bounds[1] = 1

    train_x = init_x
    train_y = init_y

    if output_dim > 1:
        bd = DominatedPartitioning(ref_point=test_function.ref_point.to(train_x), Y=train_y)
        volume = bd.compute_hypervolume().item()
        hv = torch.zeros(args["n_BO_iters"] * q + 1)
        hv[0] = volume
    
    t = time.time()
    for i in range(args["n_BO_iters"]):
        sys.stdout.flush()
        sys.stderr.flush()
        print("\niteration %d" % i)

        # fit model on normalized x
        model_start = time.time()
        normalized_x = normalize(train_x, bounds).to(train_x)
        model.fit_and_save(normalized_x, train_y, model_save_dir)
        model_end = time.time()
        print("fit time", model_end - model_start)
        
        acq_start = time.time()
        acquisition = construct_acqf_by_model(model_name, model, normalized_x, train_y, test_function)
        normalized_candidates, acqf_values = optimize_acqf(
            acquisition, standard_bounds, q=q, num_restarts=2, raw_samples=16, return_best_only=False,
            options={"batch_limit": 1, "maxiter": 10})
        candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)

        # round candiates
        candidates = round(test_function_name, candidates)
        # calculate acquisition values after rounding
        normalized_rounded_candidates = normalize(candidates, bounds)
        acqf_values = acquisition(normalized_rounded_candidates)
        acq_end = time.time()
        print("acquisition time", acq_end - acq_start)

        best_index = acqf_values.max(dim=0).indices.item()
        # best x is best acquisition value after rounding
        new_x = candidates[best_index].to(train_x)

        del acquisition
        del acqf_values
        del normalized_candidates
        del normalized_rounded_candidates
        torch.cuda.empty_cache()

        # evaluate new y values and save
        new_y = test_function(new_x)
        # add explicit output dimension
        if output_dim == 1:
            new_y = new_y.unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        
        if output_dim > 1:
            # compute hypervolume
            for h in range(q):
                bd = DominatedPartitioning(ref_point=test_function.ref_point.to(train_x), Y=train_y[:-q + h + 1])
                volume = bd.compute_hypervolume().item()
                hv[q * i + h + 1] = volume
            print("Max value", hv.max().item())
        else:
            print("Max value", train_y.max().item())

    if model_save_dir is not None:
        torch.save(train_x.cpu(), "%s/train_x.pt" % model_save_dir)
        torch.save(train_y.cpu(), "%s/train_y.pt" % model_save_dir)
        if output_dim > 1:
            torch.save(hv.cpu(), "%s/volume.pt" % model_save_dir)

    if output_dim > 1:
        hv_max_index = torch.argmax(hv)
        train_max_index = hv_max_index + len(init_x) - 1
        return train_x[train_max_index], hv[hv_max_index]
    else:
        max_index = torch.argmax(train_y)
        return train_x[max_index], train_y[max_index]


def initialize_model(model_name, model_args, input_dim, output_dim, device):
    if model_name == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    elif model_name == 'dkl':
        if output_dim == 1:
            return SingleTaskDKL(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskDKL(model_args, input_dim, output_dim, device)
    elif model_name == 'ibnn':
        if output_dim == 1:
            return SingleTaskIBNN(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskIBNN(model_args, input_dim, output_dim, device)
    elif model_name == 'hmc':
        return HMC(model_args, input_dim, output_dim, device)
    elif model_name == 'sghmc':
        return SGHMCModel(model_args, input_dim, output_dim, device)
    elif model_name == 'laplace':
        return LaplaceBNN(model_args, input_dim, output_dim, device)
    elif model_name == 'ensemble':
        return Ensemble(model_args, input_dim, output_dim, device)
    else:
        raise NotImplementedError("Model type %s does not exist" % model_name)


def initialize_points(test_function, n_init_points, output_dim, device, test_function_name):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = test_function.bounds.to(device, dtype=torch.float64)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)
        init_x = round(test_function_name, init_x)
    init_y = test_function(init_x)
    # add explicit output dimension
    if output_dim == 1:
        init_y = init_y.unsqueeze(-1)
    return init_x, init_y


def construct_acqf_by_model(model_name, model, train_x, train_y, test_function):
    sampler = StochasticSampler(sample_shape=torch.Size([128]))
    if test_function.num_objectives == 1:
        qEI = qExpectedImprovement(
            model=model,
            best_f=train_y.max(),
            sampler=sampler
        )
        return qEI
    else: # multi-objective
        with torch.no_grad():
            pred = model.posterior(train_x).mean
            pred = pred.squeeze(-1) # TODO: Laplace?
        partitioning = FastNondominatedPartitioning(
            ref_point=test_function.ref_point.to(train_x),
            Y=pred,
        )
        qEHVI = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=test_function.ref_point.to(train_x),
            partitioning=partitioning,
            sampler=sampler
        )
        return qEHVI


def get_test_function(test_function, seed):
    test_function = test_function.lower()
    if "ackley" in test_function:
        if test_function == "ackley":
            dim = 2
        else:
            dim = int(test_function.split('_')[-1])
        return Ackley(dim=dim, negate=True)
    elif test_function == "branin":
        return Branin(negate=True)
    elif test_function == "branincurrin":
        return BraninCurrin(negate=True)
    elif test_function == "hartmann":
        return Hartmann(negate=True)
    elif "dtlz1" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return DTLZ1(dim, num_objectives=obj, negate=True)
    elif "dtlz3" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return DTLZ3(dim, num_objectives=obj, negate=True)
    elif "dtlz5" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return DTLZ5(dim, num_objectives=obj, negate=True)
    elif "dtlz7" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return DTLZ7(dim, num_objectives=obj, negate=True)
    elif test_function == "oil":
        return OilSorbent(negate=True)
    elif test_function == "cco":
        return cco.CCO(negate=True)
    elif test_function == "pde":
        return PDEVar(negate=True)
    elif test_function == "lunar":
        return LunarLanderProblem()
    elif test_function == "pest":
        return PestControl(negate=True)
    elif test_function == "optics":
        return Optics()
    elif test_function == "kd":
        return KnowledgeDistillation()
    elif "bnn" in test_function:
        dim = int(test_function.split('_')[1])
        obj = int(test_function.split('_')[2])
        return BnnDraw(dim, obj, seed)
    elif "poly" in test_function:
        dim = int(test_function.split('_')[1])
        return PolyDraw(dim, seed)
    else:
        raise NotImplementedError(
            "Test function %s does not exist." % test_function)


def main(cl_args):
    
    current_time = datetime.now()
    args = json.load(open("./config/" + cl_args.config + ".json", 'r'))

    # set save_dir name
    save_dir = current_time.strftime("experiment_results/%y_%m_%d-%H_%M_%S")
    test_function_name = args["test_function"]
    if cl_args.name:
        save_dir = "%s_%s_%s" % (
            save_dir, cl_args.name, test_function_name.lower())
    else:
        save_dir = "%s_%s_%s" % (
            save_dir, cl_args.config, test_function_name.lower())
    os.makedirs(save_dir)

    try:
        if cl_args.bg:
            # redirect stdout + stderr
            sys.stdout = open(save_dir + '/stdout.txt', 'w')
            sys.stderr = open(save_dir + '/stderr.txt', 'w')

        # save config
        with open(save_dir + '/config.json', 'w') as f:
            json.dump(args, f, indent=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(int(args["seed"]))

        # get test function
        test_function = get_test_function(test_function_name, int(args["seed"]))
        input_dim = test_function.dim
        output_dim = test_function.num_objectives

        for trial in range(1, args["n_trials"] + 1):
            # torch.manual_seed(trial-1)
            # print initial info
            print("-" * 20, "START TRIAL %i" % trial, "-" * 20)
            print("Test function:", test_function.__class__.__name__)
            if hasattr(test_function, '._optimal_value'):
                print("True minimum:", test_function._optimal_value)
            
            # get initial points
            init_x, init_y = initialize_points(test_function, args["n_init_points"], output_dim, device, test_function_name)
            # run bayes opt for each model
            model_dict = args["models"]
            for model_id, model_args in model_dict.items():
                model_name = model_args["model"]

                model_save_dir = "%s/trial_%d/%s" % (save_dir, trial, model_id)
                model_state_dir = model_save_dir + "/model_state"
                os.makedirs(model_save_dir)
                os.makedirs(model_state_dir)
                os.makedirs(model_save_dir + "/queries")

                print("-" * 20, "running " + model_id, "-" * 20)
                start_time = time.time()
                model = initialize_model(model_name, model_args, input_dim, output_dim, device)
                best_x, best_y = bayes_opt(
                    model, test_function, args, init_x, init_y, model_save_dir, device, model_name, test_function_name)
                del model

                print("\nMax value found was", best_y.cpu().numpy())
                # print("at", best_x.cpu().numpy())
                print("Time(s):", time.time() - start_time)

            torch.cuda.empty_cache()


        os.rename(save_dir, save_dir + "_done")
        print("Done!")
    except:
        print(traceback.print_exc())
        os.rename(save_dir, save_dir + "_canceled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--bg", default=False, action="store_true")
    parser.add_argument("-n", "--name", type=str, help="experiment name (optional)")
    cl_args = parser.parse_args()

    main(cl_args)