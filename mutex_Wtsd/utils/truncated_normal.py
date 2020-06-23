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
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self._loc.size()
        super(TruncNorm, self).__init__(batch_shape)

        self._entrpy = None
        self._mean = None
        self._std = None
        self._var = None

        self._cap_psi_a = _norm_cdf(self._loc, self._scale, self._a)
        self._psi_a = _norm_pdf(self._loc, self._scale, self._a)
        self._cap_psi_b = _norm_cdf(self._loc, self._scale, self._b)
        self._psi_b = _norm_pdf(self._loc, self._scale, self._b)
        self._cap_z = self._cap_psi_b - self._cap_psi_a

        self._alpha = (self._a - self._loc) / self._scale
        self._beta = (self._b - self._loc) / self._scale
        self._cap_psi_alpha = _snorm_cdf(self._alpha)
        self._psi_alpha = _snorm_pdf(self._alpha)
        self._cap_psi_beta = _snorm_cdf(self._beta)
        self._psi_beta = _snorm_pdf(self._beta)


        # create variables for gradient computations
        self._loc_with_grad = Variable(self._loc.detach().clone(), requires_grad=True)
        self._scale_with_grad = Variable(self._scale.detach().clone(), requires_grad=True)
        self._cap_z_with_grad = _norm_cdf(self._loc_with_grad, self._scale_with_grad, self._b) - \
                                _norm_cdf(self._loc_with_grad, self._scale_with_grad, self._a)

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
            self._mean = self._loc - self._scale * (self._psi_beta - self._psi_alpha) / \
                         (self._cap_psi_beta - self._cap_psi_a)
        return self._mean

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        if self._var is None:
            self._var = (self._scale ** 2) * (1 - ((self._beta * self._psi_beta - self._alpha * self._psi_alpha) /
                                                   (self._cap_psi_beta - self._cap_psi_a)) -
                                              ((self._psi_beta - self._psi_alpha) /
                                               (self._cap_psi_beta - self._cap_psi_a)) ** 2)
        return self._var

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        if self._std is None:
            self._std = torch.sqrt(self.variance)
        return self._std

    def sample(self, size=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. For now this wraps a scipy function
        """
        # @TODO batch processing
        p = torch.rand_like(self._loc)
        sample = self.icdf(p)
        return sample

    def rsample(self):
        try:
            assert all(self._a < self.mean) and all(self._b > self.mean)
        except:
            j=1

        a = (self._a - self.mean.detach()) / self.stddev.detach()
        b = (self._b - self.mean.detach()) / self.stddev.detach()

        try:
            assert all(a<=b)
        except:
            j=1

        loc, scale = torch.zeros_like(self._loc), torch.ones_like(self._scale)
        p = torch.rand_like(self._loc)
        eps = self.rsmpl_icdf(loc, scale, a, b, p)
        sample = self.mean + eps * self.stddev

        try:
            assert all(sample >= self._a) and all(sample <= self._b)
        except:
            j=1

        return sample

    def log_prob(self, value):
        """
        Returns the log of the cdf function evaluated at
        value.

        Args:
            value (Tensor)
        """
        return (self.pdf(value) + 1e-40).log()
        # return torch.log(self.prob(value) + 1e-40)  # add number to prevent -inf values

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
            assert all(self._a <= value.view(-1)) and all(value.view(-1) <= self._b), "only values in support set are handled"
        return (_norm_cdf(self._loc, self._scale, value) - self._cap_psi_alpha) / self._cap_z

    def icdf(self, prob):
        """
        Returns the inverse cumulative distribution function evaluated at
        `value`.

        Args:
            prob (Tensor):
        """
        if not (torch.is_tensor(prob) or isinstance(prob, Number)):
            raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
        if isinstance(prob, Number) or prob.dim() == 0:
            assert 0 <= prob <= 1, "only values in support set are handled"
        else:
            assert all(0 <= prob.view(-1)) and all(prob.view(-1) <= 1), "only values in support set are handled"

        arg = self._cap_psi_a + prob * self._cap_z
        cd = _norm_icdf(self._loc, self._scale, arg)
        return cd

    def rsmpl_icdf(self, loc, scale, a, b, prob):
        """
        Returns the inverse cumulative distribution function evaluated at
        `value`.

        Args:
            prob (Tensor):
        """
        if not (torch.is_tensor(prob) or isinstance(prob, Number)):
            raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
        if isinstance(prob, Number) or prob.dim() == 0:
            assert 0 <= prob <= 1, "only values in support set are handled"
        else:
            assert all(0 <= prob.view(-1)) and all(prob.view(-1) <= 1), "only values in support set are handled"

        arg = _norm_cdf(loc, scale, a) + prob * (_norm_cdf(loc, scale, b) - _norm_cdf(loc, scale, a))
        cd = _norm_icdf(loc, scale, arg)
        return cd

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
            assert all(self._a <= value.view(-1)) and all(value.view(-1) <= self._b), "only values in support set are handled"

        return _norm_pdf(self._loc, self._scale, value) / self._cap_z

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
            assert all(self._a <= value.view(-1)) and all(value.view(-1) <= self._b), "only values in support set are handled"

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

    def plot_funcs(self, idx):
        import matplotlib.pyplot as plt
        range_list = torch.tensor(list(range(0, 100))).float().to(self._loc.device) / 100

        vals = torch.stack([self.pdf(i).detach().cpu() for i in range_list], dim=-1)
        plt.plot(range_list.cpu(), vals[idx], 'r-')
        plt.ylabel('truncnorm pdf')
        plt.show()

        vals = torch.stack([self.cdf(i).detach().cpu() for i in range_list], dim=-1)
        plt.plot(range_list.cpu(), vals[idx], 'r-')
        plt.ylabel('truncnorm cdf')
        plt.show()



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    x = Variable(torch.arange(-50, 50, dtype=torch.float) / 20, requires_grad=True)
    erfval = torch.erf(x)
    grad_erf = grad(erfval.sum(), x)

    range_list = x.detach()
    # plt.plot(range_list, erfval.detach(), 'r-')
    # plt.ylabel('erf(x)')
    # plt.show()
    # plt.plot(range_list, grad_erf[0], 'b-')
    # plt.ylabel('derf(x) / dx')
    # plt.show()

    means = torch.tensor([.5, 0.64, 0.24]).float()
    sigmas = torch.ones_like(means).float() / 10
    dis = TruncNorm(means, sigmas, 0, 1, eval_range=0.05)

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

    range_list = torch.tensor(list(range(0, 100))).float().to(dis._loc.device) / 100

    vals_pdf = torch.stack([dis.pdf(i).detach().cpu() for i in range_list], dim=-1)
    vals_cdf = torch.stack([dis.cdf(i).detach().cpu() for i in range_list], dim=-1)
    vals_pdf_scipy = torch.stack([torch.from_numpy(dis._rvs.pdf(i)) for i in range_list], dim=-1)
    vals_cdf_scipy = torch.stack([torch.from_numpy(dis._rvs.cdf(i)) for i in range_list], dim=-1)

    for pdf, cdf, pdf_scipy, cdf_scipy in zip(vals_pdf, vals_cdf, vals_pdf_scipy, vals_cdf_scipy):
        plt.plot(range_list.detach().cpu(), pdf, 'r-')
        plt.ylabel('my truncnorm pdf')
        plt.show()
        plt.plot(range_list.detach().cpu(), pdf_scipy, 'b-')
        plt.ylabel('scipy truncnorm pdf')
        plt.show()

        plt.plot(range_list.detach().cpu(), cdf, 'r-')
        plt.ylabel('my truncnorm cdf')
        plt.show()
        plt.plot(range_list.detach().cpu(), cdf_scipy, 'b-')
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
