
import torch
from scipy.stats import levy_stable
import numpy as np
from functools import partial


def gaussian_pdf(x, mu=0, sigma=torch.sqrt(torch.tensor(2))):
    return 1 / (sigma * torch.sqrt(torch.tensor(2*torch.pi))) * torch.exp(-1/2 * ((x-mu) / sigma) ** 2)

def gaussian_score(x, mu=0, sigma=torch.sqrt(torch.tensor(2))):
    return (mu - x) / (sigma ** 2)

def score_fourier_transform(x: torch.Tensor, alpha):
    """
    get alpha stable score through fourier transform
    unfortunately, not working properly
    """

    N = 100
    diff_q = torch.tensor(0, dtype=torch.complex128)
    q = torch.tensor(0, dtype=torch.complex128)
    i = torch.complex(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64))

    for t in range(0, N):
        tmp = torch.exp(-(torch.abs(torch.tensor(2 * torch.pi * t / N)) ** alpha) - i * 2 * torch.pi * x * t / N)
        diff_q += -2 * torch.pi * i * t / N * tmp
        q += tmp

    return diff_q / q

def score_finite_diff(x, alpha):
    """
    calculate alpha stable score through finite difference
    """

    def grad_approx(func, x, h=0.001):
        # https://arxiv.org/abs/1607.04247
        if type(x) is np.ndarray and type(h) is not np.ndarray:
            h = np.array(h)

        return (-func(x + 2 * h) + 8 * func(x + h) - 8 * func(x - h) + func(x - 2 * h)) / (12 * h)

    levy_logpdf = partial(levy_stable.logpdf, alpha=alpha, beta=0)

    return grad_approx(levy_logpdf, x)