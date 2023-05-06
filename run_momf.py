import json
import os
import time

import numpy as np
import pandas as pd
import torch

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize

from gpytorch import ExactMarginalLogLikelihood

from bo import optimize_MOMF

from problems.utils import get_problem, scale_x


# Set dataset and algorithm name
dataset_name = "GERMANCREDIT"
algo_name = "XGB"

# Load the budget from json
ntrial = 10
budget = json.load(open("data/config/budget.json", "r"))[algo_name][dataset_name]

# Read datasets, set sensitive features, hyperparameters and target
dataset_full = pd.read_csv(f"data/{dataset_name}_full.txt", sep=",")
dataset_half = pd.read_csv(f"data/{dataset_name}_redux.txt", sep=",")
dataset_info = json.load(open("data/config/dataset_info.json", "r"))[dataset_name]
sensitive_features = dataset_info["sensitive"]
target = dataset_info["target"]
algo_info = json.load(open("data/config/algorithm_info.json", "r"))[algo_name]
hp = algo_info["hyperparameters"]

# Initialize the problem
problem = get_problem(algo_name)(dataset_full, dataset_half, target, sensitive_features, algo_info["seed"])

print("------------------------------")
print(f"DATASET: {dataset_name}")
print(f"ALGORITHM: {algo_name}")
print("------------------------------")

# Loop over different trial
for trial in range(1, ntrial + 1):
    print(f"\nTrial {trial} out of 10...")

    # Set seed and budget
    torch.manual_seed(trial)
    np.random.seed(trial)
    current_budget = budget[str(trial)]

    # Create the results folder
    os.makedirs(f"result/{algo_name}_HPO_{dataset_name}_results", exist_ok=True)

    # Get initial design
    print("Initial design...", end=" ")
    init_design = pd.read_csv(f"data/initial_design/"
                              f"{algo_name.upper()}_HPO_{dataset_name.upper()}_init/"
                              f"init_design_{trial}.csv")
    train_x = torch.tensor(init_design[hp + ["source"]].values).double()
    train_x = scale_x(train_x, algo_name)
    # Note: BoTorch assumes a maximization problem
    train_obj = 1 - torch.tensor(init_design[["mcs", "dsp"]].values).double()
    cumulated_cost = torch.sum((train_x[:, -1] + 1) / 2)
    print(f"[c={cumulated_cost}]")

    # Initialize list to save computational times
    model_update_time = [0.] * train_x.shape[0]
    acquisition_time = [0.] * train_x.shape[0]
    query_time = [0.] * train_x.shape[0]

    # BO loop
    while cumulated_cost < current_budget:

        # Fit the GP
        start_time = time.time()
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model_update_time.append(time.time() - start_time)

        # Optimize the acquisition
        start_time = time.time()
        new_x = optimize_MOMF(problem, model, train_obj)
        acquisition_time.append(time.time() - start_time)
        new_x[:, -1] = torch.round(new_x[:, -1], decimals=0)

        # New observation and update dataset
        start_time = time.time()
        new_obj = problem(new_x)
        query_time.append(time.time() - start_time)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        cumulated_cost = torch.sum((train_x[:, -1] + 1) / 2)

        print(f"Budget: {cumulated_cost} out of {current_budget}")

    # Save the results for the current trial in a csv file
    res = pd.DataFrame(train_x, columns=hp + ["source"])
    res["msc"] = 1 - train_obj[:, 0]
    res["dsp"] = 1 - train_obj[:, 1]
    res["model_update_time"] = model_update_time
    res["acquisition_time"] = acquisition_time
    res["query_time"] = query_time

    res.to_csv(f"result/{algo_name}_HPO_{dataset_name}_results/result_{trial}.csv", index=False)
