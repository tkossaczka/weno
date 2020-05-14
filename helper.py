import torch
import numpy as np

def compute_combinations(u,lag):
    lagged_us = [u[k:k-lag] for k in range(lag)]
    lagged_us.append(u[lag:])
    pairs = []
    for k in range(lag+1):
        for l in range(k+1):
            pairs.append(lagged_us[l] * lagged_us[k])
    comb5_bn0 = pairs[:6]
    comb5_bn1 = [pairs[i] for i in [2,4,5,7,8,9]]
    comb5_bn2 = [pairs[i] for i in [5,8,9,12,13,14]]
    comb5_bp0 = [pairs[i] for i in [2,4,5,7,8,9]]
    comb5_bp1 = [pairs[i] for i in [5,8,9,12,13,14]]
    comb5_bp2 = [pairs[i] for i in [9,13,14,18,19,20]]

    comb6_bn0 = pairs[:10]
    comb6_bn1 = [pairs[i] for i in [2,4,5,7,8,9,11,12,13,14]]
    comb6_bn2 = [pairs[i] for i in [5,8,9,12,13,14,17,18,19,20]]
    comb6_bp0 = [pairs[i] for i in [2,4,5,7,8,9,11,12,13,14]]
    comb6_bp1 = [pairs[i] for i in [5,8,9,12,13,14,17,18,19,20]]
    comb6_bp2 = [pairs[i] for i in [9,13,14,18,19,20,24,25,26,27]]

    listisko5 =   [torch.stack(comb5_bn0, dim=1), torch.stack(comb5_bn1, dim=1), torch.stack(comb5_bn2, dim=1), torch.stack(comb5_bp0, dim=1),
           torch.stack(comb5_bp1, dim=1), torch.stack(comb5_bp2, dim=1)]
    listisko6  = [torch.stack(comb6_bn0, dim=1),
           torch.stack(comb6_bn1, dim=1), torch.stack(comb6_bn2, dim=1), torch.stack(comb6_bp0, dim=1),
           torch.stack(comb6_bp1, dim=1), torch.stack(comb6_bp2, dim=1)]

    return listisko5, listisko6

