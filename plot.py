import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--mo", default=False, action="store_true")
f_args = parser.parse_args()

dir = f_args.dir + "/"
multi_obj = f_args.mo

# read arguments from config
args = json.load(open(dir + "config.json", 'r'))
n_trials = args["n_trials"]
test_function = args["test_function"]
init_points = args["n_init_points"]

plt.figure(figsize=(8, 6))
for i, (model_id, model_args) in enumerate(args["models"].items()):
    # store maximum reward
    max_reward = torch.tensor([])
    for t in range(1, n_trials + 1):
        model_dir = dir + ("trial_%d/" % t) + model_id + "/"
        if multi_obj:
            train_y = torch.load(model_dir + "volume.pt")
        else:
            train_y = torch.load(model_dir + "train_y.pt")

        max_trial_reward = torch.zeros_like(train_y)
        for i in range(len(max_trial_reward)):
            max_trial_reward[i] = train_y[:(i+1)].max()

        max_trial_reward = max_trial_reward[init_points:]
        max_reward = torch.cat((max_reward, max_trial_reward.unsqueeze(0)))

    reward_mean = max_reward.mean(dim=0)
    reward_std_error = max_reward.std(dim=0, unbiased=False) / np.sqrt(n_trials)

    xs = range(init_points, len(reward_mean) + init_points)
    plt.plot(xs, reward_mean, label=model_id, linewidth=4)
    plt.fill_between(xs, (reward_mean-reward_std_error), (reward_mean+reward_std_error), alpha=0.1)

plt.legend()
plt.title(test_function)
plt.xlabel("Function Evaluations")
plt.ylabel("Max Reward")
plt.tight_layout()
# plt.savefig(dir + test_function + "_plot.pdf", bbox_inches="tight")
plt.savefig(dir + test_function + "_plot.png", bbox_inches="tight")