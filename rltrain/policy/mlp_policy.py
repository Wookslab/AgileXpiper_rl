# policy/mlp_policy.py

import torch
import torch.nn as nn
from .base_policy import BasePolicy

class MLPPolicy(nn.Module, BasePolicy):
    def __init__(self, obs_dim, act_dim):
        super(MLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs):
        with torch.no_grad():
            out = self.forward(obs)
            if torch.isnan(out).any():
                print("[ERROR] NaN detected in policy output!")
            return out.cpu().numpy()