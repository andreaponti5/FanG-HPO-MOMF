import numpy as np
import importlib
import torch


def get_problem(algorithm):
    module_name = "problems.hpo_" + algorithm.lower()
    problem_name = "Hpo" + algorithm.capitalize()
    mod = importlib.import_module(module_name)
    return getattr(mod, problem_name)


def scale_x(train_x, algorithm):
    scaled_x = globals()[f"scale_x_{algorithm.lower()}"](train_x)
    scaled_x[:, -1] = 1 - (scaled_x[:, -1] - 1)
    return scaled_x


def scale_x_mlp(train_x):
    for i in range(5, 10):
        train_x[:, i] = torch.pow(10, train_x[:, i])
    return train_x


def scale_x_rf(train_x):
    return train_x


def scale_x_xgb(train_x):
    for i in [1, 3, 4]:
        train_x[:, i] = torch.pow(10, train_x[:, i])
    return train_x


def statistical_parity_difference(y_pred, sensitive_features):
    prob_y_given_s = []
    N = y_pred.shape[0]
    prob_y = np.where(y_pred == 1)[0].shape[0] / N
    for value in [0, 1]:
        idx = np.where(sensitive_features == value)[0]
        if idx.shape[0] == 0:
            return 0
        prob_s = idx.shape[0] / N
        prob_y_s = np.where((y_pred == 1) | (sensitive_features == value))[0].shape[0] / N
        prob_y_given_s.append((prob_y + prob_s - prob_y_s) / prob_s)
    return abs(prob_y_given_s[0] - prob_y_given_s[1])


def unscale_x(train_x, algorithm):
    scaled_x = globals()[f"unscale_x_{algorithm.lower()}"](train_x)
    scaled_x[:, -1] = 1 - scaled_x[:, -1] + 1
    return scaled_x


def unscale_x_mlp(train_x):
    for i in range(5, 10):
        train_x[:, i] = torch.log10(train_x[:, i])
    return train_x


def unscale_x_rf(train_x):
    return train_x


def unscale_x_xgb(train_x):
    for i in [0, 6]:
        train_x[:, i] = torch.round(train_x[:, i], decimals=0)
    for i in [1, 3, 4]:
        train_x[:, i] = torch.log10(train_x[:, i])
    return train_x
