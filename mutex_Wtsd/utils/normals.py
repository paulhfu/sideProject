import torch
import math
import numpy as np

def _snorm_cdf(value):
    """cdf of standard normal"""
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))

def _snorm_pdf(value):
    """pdf of standard normal"""
    return torch.exp((-(value ** 2)) / 2) / (np.sqrt(2 * np.pi))

def _norm_cdf(mu, sig, value):
    """cdf of normal"""
    return 0.5 * (1 + torch.erf((value - mu) / (math.sqrt(2) * sig)))

def _norm_icdf(mu, sig, prob):
    """inverse cdf of normal"""
    return mu + sig * torch.erfinv(2 * prob - 1) * math.sqrt(2)

def _norm_pdf(mu, sig, value):
    """pdf of normal"""
    return torch.exp((-(((value - mu) / sig) ** 2)) / 2) / (sig * np.sqrt(2 * np.pi))