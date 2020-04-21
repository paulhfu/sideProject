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

class TruncNorm(Distribution):
    """
    mostly from https://en.wikipedia.org/wiki/Truncated_normal_distribution
    and https://people.sc.fsu.edu/~%20jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive,
                       'a': constraints.real, 'b': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0
    def __init__(self, loc, scale, a, b, eval_range):
        """
        loc and scale are the respective parameters for the final truncated distribution NOT the ones of the parent
        distribution
        :param loc: mean (mu)
        :param scale: standard deviation (sigma)
        :param a: lower end of truncation
        :param b: upper end of truncation
        """
        # assert all(scale > 0)
        assert a < b
        # assert all(scale >= a) and all(scale <= b)
        self._loc, self._scale = broadcast_all(loc, scale)
        self._a, self._b = a, b
        self._eval_range = eval_range
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self._loc.size()
        super(TruncNorm, self).__init__(batch_shape, 1)

        # do the transformations
        self._alpha, self._beta = (a - self._loc) / self._scale, (b - self._loc) / self._scale

        # keep this for sampling
        self._rvs = truncnorm(self._alpha.detach().cpu().numpy(), self._beta.detach().cpu().numpy(),
                              loc=self._loc.detach().cpu().numpy(), scale=self._scale.detach().cpu().numpy())


        self._entrpy = None
        self._mean = None
        self._std = None
        self._var = None

        self._cap_psi_alpha = _snorm_cdf(self._alpha)
        self._psi_alpha = _snorm_pdf(self._alpha)
        self._psi_beta = _snorm_pdf(self._beta)
        self._cap_z = _snorm_cdf(self._beta) - self._cap_psi_alpha

        # create variables for gradient computations
        self._loc_with_grad = Variable(self._loc.detach().clone(), requires_grad=True)
        self._alpha_with_grad, self._beta_with_grad = (a - self._loc_with_grad) / self._scale, (b - self._loc_with_grad) / self._scale
        self._cap_z_with_grad = _snorm_cdf(self._beta_with_grad) - _snorm_cdf(self._alpha_with_grad)

    def _zeta(self, value):
        return (value - self._loc) / self._scale

    def _zeta_with_grad(self, value):
        return (value - self._loc_with_grad) / self._scale

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        # TODO implement instance copying
        pass

    @property
    def mean_prob(self):
        return

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        if self._mean is None:
            self._mean = self._loc + self._scale * (self._psi_alpha - self._psi_beta) / self._cap_z
        return self._mean

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        if self._var is None:
            self._var = (self._scale ** 2) * (1 + ((self._psi_alpha - self._psi_beta) / self._cap_z) -
                                              ((self._psi_alpha - self._psi_beta) / self._cap_z) ** 2)
        return self._var

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        if self._var is None:
            self._var = self._scale ** 2 (1 + ((self._psi_alpha - self._psi_beta) / self._cap_z) -
                                          ((self._psi_alpha - self._psi_beta) / self._cap_z) ** 2)
        return torch.sqrt(self._var)

    def sample(self, size=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. For now this wraps a scipy function
        """
        # TODO implement https://arxiv.org/pdf/1201.6140.pdf
        #  (FAST  SIMULATION  OF  TRUNCATED  GAUSSIANDISTRIBUTIONS(NICOLAS CHOPIN, ENSAE-CREST))

        return torch.as_tensor(self._rvs.rvs(), device=self._loc.device, dtype=self._loc.dtype)

    def rsample(self, sample_shape=torch.Size()):
        # TODO this is actually for non-trunc normal...
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self._loc.dtype, device=self._loc.device)
        sample = self.mean + eps * torch.sqrt(self.variance)
        while not all(self._a <= sample <= self._b):
            sample = self.mean + eps * torch.sqrt(self.variance)
        return sample

    def prob(self, value):
        """
        Returns the cdf function evaluated at
        `value2` - 'value1'.

        Args:
            value1 (Tensor):
            value2 (Tensor):
        """
        if not (torch.is_tensor(value) or isinstance(value, Number)):
            raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
        if isinstance(value, Number) or value.dim() == 0:
            assert self._a <= value <= self._b, "only values in support set are handled"
        else:
            assert all(self._a <= value) and all(value <= self._b), "only values in support set are handled"

        value1, value2 = (value - self._eval_range).clamp(max=self._b, min=self._a), \
                         (value + self._eval_range).clamp(max=self._b, min=self._a)
        return self.cdf(value2) - self.cdf(value1)


    def log_prob(self, value):
        """
        Returns the log of the cdf function evaluated at
        `value2` - 'value1'.

        Args:
            value1 (Tensor):
            value2 (Tensor):
        """
        return torch.log(self.prob(value) + 1e-40)  # add number to prevent -inf values

    def cdf(self, value):
        """
        Returns the cumulative distribution function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        if not (torch.is_tensor(value) or isinstance(value, Number)):
            raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
        if isinstance(value, Number) or value.dim() == 0:
            assert self._a <= value <= self._b, "only values in support set are handled"
        else:
            assert all(self._a <= value) and all(value <= self._b), "only values in support set are handled"

        return (_snorm_cdf(value) - self._cap_psi_alpha) / self._cap_z

    def pdf(self, value):
        """
        Returns the density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        if not (torch.is_tensor(value) or isinstance(value, Number)):
            raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
        if isinstance(value, Number) or value.dim() == 0:
            assert self._a <= value <= self._b, "only values in support set are handled"
        else:
            assert all(self._a <= value) and all(value <= self._b), "only values in support set are handled"

        value = self._zeta(value)
        return _snorm_pdf(value) / (self._cap_z * self._scale).detach()

    def grad_pdf_mu(self, value):
        """
        returns the grad of the pdf w.r.t. mu at value
        Args:
            value (Tensor):
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self._loc.dtype, device=self._loc.device)
        if value.dim() == 0:
            assert self._a <= value <= self._b, "only values in support set are handled"
        else:
            assert all(self._a <= value) and all(value <= self._b), "only values in support set are handled"

        value = self._zeta_with_grad(value)
        f_val = _snorm_pdf(value) / (self._cap_z_with_grad)
        grad_f = grad(f_val.sum(), self._loc_with_grad, retain_graph=True)
        return grad_f[0]

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        if self._entrpy is None:
            self._entrpy = torch.log(np.sqrt(2 * np.pi * np.e) * self._scale * self._cap_z) \
                  + (self._alpha * self._psi_alpha - self._beta*self._psi_alpha) / (2 * self._cap_z)
        raise self._entrpy


def _snorm_cdf(value):
    """cdf of standard normal"""
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))

def _snorm_pdf(value):
    """pdf of standard normal"""
    return torch.exp((-(value ** 2)) / 2) / (np.sqrt(2 * np.pi))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    x = Variable(torch.arange(-50, 50, dtype=torch.float) / 20, requires_grad=True)
    erfval = torch.erf(x)
    grad_erf = grad(erfval.sum(), x)

    range_list = x.detach()
    plt.plot(range_list, erfval.detach(), 'r-')
    plt.ylabel('erf(x)')
    plt.show()
    plt.plot(range_list, grad_erf[0], 'b-')
    plt.ylabel('derf(x) / dx')
    plt.show()

    means = torch.tensor(.5).float()
    sigmas = torch.ones_like(means)
    dis = TruncNorm(means, sigmas, 0, 1)

    grad1 = dis.grad_pdf_mu(0.5)
    grad2 = dis.grad_pdf_mu(0.2)
    grad3 = dis.grad_pdf_mu(0.7)

    s1 = dis.sample(1)
    s2 = dis.sample(2)
    s3 = dis.sample(3)

    means = dis.mean
    var = dis.variance

    logproba1 = dis.log_prob(.3)
    logproba2 = dis.log_prob(.3)
    logproba3 = dis.log_prob(.3)

    range_list = torch.tensor(list(range(0, 100))).float() / 100
    plt.plot(range_list, dis.pdf(range_list), 'r-')
    plt.ylabel('my truncnorm pdf')
    plt.show()
    plt.plot(range_list, dis._rvs.pdf(range_list), 'b-')
    plt.ylabel('scipy truncnorm pdf')
    plt.show()

    plt.plot(range_list, dis.cdf(range_list), 'r-')
    plt.ylabel('my truncnorm cdf')
    plt.show()
    plt.plot(range_list, dis._rvs.cdf(range_list), 'b-')
    plt.ylabel('scipy truncnorm cdf')
    plt.show()

    pdf_val1 = dis.pdf(.3)
    comp1 = np.abs((pdf_val1.numpy() - dis._rvs.pdf(.3))).max()
    pdf_val2 = dis.pdf(.2)
    comp2 = np.abs((pdf_val2.numpy() - dis._rvs.pdf(.2))).max()

    cdf_val1 = dis.cdf(.3)
    comp3 = np.abs(cdf_val1.numpy() - dis._rvs.cdf(.3)).max()
    cdf_val2 = dis.cdf(.2)
    comp4 = np.abs(cdf_val2.numpy() - dis._rvs.cdf(.2)).max()

    assert comp1 + comp2 < 1e-6, "pdf values differ"
    assert comp3 + comp4 < 1e-6, "cdf values differ"
    #
    # a = torch.ones(100, requires_grad=True) * 3
    # b = Variable(torch.ones(100, requires_grad=True) * 5, requires_grad=True)
    # c = Variable(torch.ones(100, requires_grad=True) * 0.3, requires_grad=True)
    #
    # soln = a * b + c
    #
    # grad_b = grad(soln.sum(), b)
    # grad_c = grad(soln.sum(), c)
    a=1
