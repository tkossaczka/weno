import torch
import numpy as np

def autocor_combinations(u, lag, return_lags=False, return_pairs=True):
    lagged_us = [u[k:k-lag] for k in range(lag)]
    lagged_us.append(u[lag:])
    multiplied_pairs=[]
    for k in range(lag+1):
        for l in range(k+1):
            multiplied_pairs.append(lagged_us[l] * lagged_us[k])

    output_list = return_lags * lagged_us + return_pairs * multiplied_pairs
    return torch.stack(output_list, dim=1)

if __name__=="__main__":
    u = torch.Tensor(np.arange(10, dtype=float))
    c = autocor_combinations(u, lag=5, return_lags=False, return_pairs=True)
    c_np = np.array(c)







