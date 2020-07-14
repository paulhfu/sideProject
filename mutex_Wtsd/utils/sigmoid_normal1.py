import torch
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import Normal
import math

class SigmNorm(TransformedDistribution):

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = Normal(loc, scale)
        transforms = [SigmoidTransform()]
        self._mean = None
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        if self._mean is None:
            self._mean = torch.sigmoid(self.loc / (torch.sqrt(1 + math.pi * self.scale ** 2 / 8)))
        return self._mean

    def rsample(self, sample_shape=torch.Size()):
        sample = super(SigmNorm, self).rsample(sample_shape)
        return sample


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    print(sys.float_info)
    loc = torch.ones(100, dtype=torch.float) * 100
    scale = (torch.arange(100, dtype=loc.dtype) + 1) / 30
    dist = SigmNorm(loc, scale)
    samples = dist.rsample()
    lp = dist.log_prob(samples)
    a=1

