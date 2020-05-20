#from define_BS_WENO import WENONetwork
import torch

train_model = torch.load("model")

params={'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'm': 160}
train_model.compare_wenos(params=params)
err_t, order_t = train_model.order_compute(params=params, mm=20, trainable=True)
err_n, order_n = train_model.order_compute(params=params, mm=20, trainable=False)
