from define_WENO_Network import WENONetwork

import torch
from torch import nn

class LargeWENONetwork(WENONetwork):
    def get_inner_nn_weno5(self):
        net = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(12, 3, kernel_size=7, stride=1, padding=2),
            nn.Sigmoid())
        return net

    def get_inner_nn_weno6(self):
        net = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(12, 3, kernel_size=7, stride=1, padding=2),
            nn.Sigmoid())
        return net