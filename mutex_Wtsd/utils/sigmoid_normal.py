import torch
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions import Normal
import math
from numbers import Number
from scipy.stats import truncnorm
from torch.autograd import grad
from torch.autograd import Variable
import numpy as np
from torch.distributions.utils import _standard_normal, broadcast_all
from utils.normals import _norm_cdf, _snorm_pdf, _norm_icdf, _norm_pdf, _snorm_cdf

class SigmNorm(Distribution):
    """
    mostly from https://en.wikipedia.org/wiki/Truncated_normal_distribution
    and https://people.sc.fsu.edu/~%20jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0
    def __init__(self, loc, scale):
        """
        loc and scale are the respective parameters for the final truncated distribution NOT the ones of the parent
        distribution
        :param loc: mean (mu)
        :param scale: standard deviation (sigma)
        """
        self._loc, self._scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self._loc.size()
        super(SigmNorm, self).__init__(batch_shape)
        self._parent_normal = Normal(self._loc, self._scale)
        self._mean = None

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        if self._mean is None:
            self._mean = torch.sigmoid(self._loc / (torch.sqrt(1 + math.pi * self._scale ** 2 / 8)))
        return self._mean

    def sample(self, size=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. For now this wraps a scipy function
        """
        size = size + self._loc.size()
        sample = torch.sigmoid(self._parent_normal.sample(size))
        return sample

    def rsample(self, size=torch.Size()):

        size = size + self._loc.size()
        eps = torch.randn(size, dtype=self._loc.dtype, device=self._loc.device)
        sample = torch.sigmoid(self._loc + eps * self._scale)

        return sample

    def log_prob(self, value):
        """
        Returns the log of the cdf function evaluated at
        value.

        Args:
            value (Tensor)
        """
        denom = (1 - value) + np.finfo(float).eps  # prevent 0-div
        exp_val = value / denom
        parent_val = torch.log(exp_val)  # inverse sigmoid
        log_prob = self._parent_normal.log_prob(parent_val)
        return log_prob


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

