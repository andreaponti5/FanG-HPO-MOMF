import torch

from botorch.acquisition.multi_objective import MOMF
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning


BATCH_SIZE = 1
MC_SAMPLES = 128
NUM_RESTARTS = 10
RAW_SAMPLES = 512


def cost_function(x):
    return torch.sum(torch.round(x[:, :, -1].flatten(), decimals=0) + 1)


def optimize_MOMF(problem, model, train_obj):
    acq_func = MOMF(
        model=model,
        ref_point=problem.ref_point,
        partitioning=FastNondominatedPartitioning(ref_point=problem.ref_point, Y=train_obj),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        cost_call=cost_function,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor(problem._bounds).T,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True}
    )
    new_x = candidates.detach()
    return new_x
