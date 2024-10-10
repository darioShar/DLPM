import torch
from functools import lru_cache
from .levy import LevyStable
import numpy as np
from .score_hpo_result import rectified_hpo_result, real_linear_hpo_result, exponent_alpha_related_result


def get_approx_score(x, alpha, is_mid_real_score=True):
    if alpha == 2:
        return - x / 2

    extreme_pts, c, t = _get_c_t(alpha)

    approx_score = (-torch.sign(x) * c * (x ** t))

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score


@lru_cache()
def _get_c_t(alpha):
    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    if len(extreme_pts) != 2:
        idx = len(extreme_pts) // 2
        extreme_pts = [-extreme_pts[idx], extreme_pts[idx]]

    x = torch.linspace(0, extreme_pts[-1].item(), 100)
    t = torch.linspace(0.1, 1, 1000).reshape(-1, 1)
    c = torch.linspace(0.1, 10, 1000).reshape(-1, 1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=2)

    t_idx = torch.argmin(res) % 1000
    c_idx = torch.argmin(res) // 1000

    return extreme_pts, c.ravel()[c_idx], t.ravel()[t_idx]


def get_approx_score2(x, alpha, is_mid_real_score=True):
    if alpha == 2:
        return - x / 2

    t = alpha * 0.5
    extreme_pts, c, = _get_c(alpha, t)
    approx_score = torch.zeros_like(x, dtype=torch.float32)
    approx_score[x >= 0] = (-c * (x ** t))[x >= 0]
    approx_score[x < 0] = (c * ((-x) ** t))[x < 0]

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score


def get_approx_score3(x, alpha, is_mid_real_score=True):
    if alpha == 2:
        return - x / 2

    t = alpha * 0.5 - 0.1
    extreme_pts, c, = _get_c(alpha, t)
    approx_score = torch.zeros_like(x, dtype=torch.float32)
    approx_score[x >= 0] = (-c * (x ** t))[x >= 0]
    approx_score[x < 0] = (c * ((-x) ** t))[x < 0]

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score


@lru_cache()
def _get_c(alpha, t):
    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    x = torch.linspace(0, extreme_pts[1].item(), 100)
    c = torch.linspace(0.01, 10, 1000).reshape(-1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=-1)

    c_idx = torch.argmin(res)

    return extreme_pts, c.ravel()[c_idx]


def get_extreme_pts(func):
    x = torch.arange(-10, 10, 0.01)
    y = func(x)
    dy = y[1:] - y[:-1]
    indice = ((dy[1:] * dy[:-1]) <= 0).nonzero()

    # remove duplicate
    new_indice = []
    for i in range(len(indice) - 1):
        if indice[i] + 1 == indice[i + 1]:
            continue
        else:
            new_indice.append(indice[i])

    new_indice.append(indice[-1])
    new_indice = torch.Tensor(new_indice).long()

    return x[new_indice + 1]


def rectified_tuning_score(x, alpha):
    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    if alpha not in alphas:
        raise NotImplementedError()

    c_hat = rectified_hpo_result[alpha]["c_hat"]
    beta_hat = rectified_hpo_result[alpha]["beta_hat"]
    score = - torch.sign(x) * c_hat * torch.abs(x) ** beta_hat
    return score


def real_linear_tuning_score(x, alpha):
    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    if alpha not in alphas:
        raise NotImplementedError()

    score_coeff1 = real_linear_hpo_result[alpha]['score_coeff1']
    score_coeff2 = real_linear_hpo_result[alpha]['score_coeff2']
    levy = LevyStable()
    real_score = levy.score(alpha=alpha, x=x)
    linear_score = -1 / 2 * x
    score = score_coeff1 * real_score + score_coeff2 * linear_score
    return score


def fitting_gen_gaussian_score(x, alpha):
    def gen_gaussian_score(x, beta, scale=1):
        return - beta * torch.sign(x) * torch.abs(x / scale) ** (beta - 1) / scale

    return gen_gaussian_score(x, beta=alpha, scale=2 ** ((alpha + 1) / 3))


def generalized_gaussian_score(x, alpha):
    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    if alpha not in alphas:
        raise NotImplementedError()

    c_hat = exponent_alpha_related_result[alpha]["c_hat"]
    beta_hat = exponent_alpha_related_result[alpha]["beta_hat"]
    score = - torch.sign(x) * c_hat * torch.abs(x) ** beta_hat
    return score
