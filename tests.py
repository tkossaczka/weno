from Run_BS_WENO import WENONetwork
from BS_WENO import BS_WENO
import numpy as np
import matplotlib.pyplot as plt
import torch

train_model=WENONetwork()
params = train_model.get_params()
V, S, tt = BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],params["xr"], params["m"], train_model.weights5, train_model.weights6)
plt.plot(S, V.detach().numpy())

V, S, tt = BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"], params["xr"], params["m"], torch.zeros([6,6]), torch.zeros([6+4,6]))
plt.plot(S, V.detach().numpy())



