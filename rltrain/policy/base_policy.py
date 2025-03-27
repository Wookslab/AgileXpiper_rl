# policy/base_policy.py

from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def forward(self, obs):
        """Forward pass: obs (torch.Tensor) -> action (torch.Tensor)"""
        pass

    @abstractmethod
    def act(self, obs):
        """Returns action as numpy array for given observation"""
        pass

    @abstractmethod
    def parameters(self):
        """Returns model parameters for optimization"""
        pass
