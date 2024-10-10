import torch
from Cython import inline
from torchquad import Simpson, MonteCarlo  # The available integrators
from torch.distributions.exponential import Exponential
# from torchlevy import LevyGaussian
from .torch_dictionary import TorchDictionary
# from .util import gaussian_score
from functools import lru_cache

import math

class LevyStable:
    def pdf(self, x: torch.Tensor, alpha, beta=0, is_cache=False):
        """
            calculate pdf through zolotarev thm
            ref. page 7, https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2894444_code545.pdf?abstractid=2894444&mirid=1
        """
        if is_cache:
            dense_dict, large_dict = _get_pdf_dict(alpha)
            ret = dense_dict.get(x)
            ret[abs(x) > 10] = large_dict.get(x)[abs(x) > 10]
            return ret

        x_flatten = x.reshape((-1))

        if alpha > 1 and beta == 0:
            ret = self._pdf_simple(x_flatten, alpha)
            return ret.reshape(x.shape)
        else:
            ret = self._pdf(x_flatten, alpha, beta)
            return ret.reshape(x.shape)

    def _pdf(self, x: torch.Tensor, alpha, beta=0):

        pi = torch.tensor(torch.pi, dtype=torch.float64)
        zeta = -beta * torch.tan(pi * alpha / 2.)

        if alpha != 1:
            x0 = x + zeta  # convert to S_0 parameterization
            xi = torch.arctan(-zeta) / alpha

            if x0 == zeta:
                gamma_func = lambda a: torch.exp(torch.special.gammaln(a))
                alpha = torch.tensor(alpha)
                alpha.requires_grad_()

                return gamma_func(1 + 1 / alpha) * torch.cos(xi) / torch.pi / ((1 + zeta ** 2) ** (1 / alpha / 2))

            elif x0 > zeta:

                def V(theta):
                    return torch.cos(alpha * xi) ** (1 / (alpha - 1)) * \
                           (torch.cos(theta) / torch.sin(alpha * (xi + theta))) ** (alpha / (alpha - 1)) * \
                           (torch.cos(alpha * xi + (alpha - 1) * theta) / torch.cos(theta))

                @inline
                def g(theta):
                    return V(theta) * (x0 - zeta) ** (alpha / (alpha - 1))

                @inline
                def f(theta):
                    g_ret = g(theta)
                    g_ret = torch.nan_to_num(g_ret, posinf=0, neginf=0)

                    return g_ret * torch.exp(-g_ret)

                # spare calculating integral on null set
                # use isclose as macos has fp differences
                if torch.isclose(-xi, pi / 2, rtol=1e-014, atol=1e-014):
                    return 0.

                simp = Simpson()
                intg = simp.integrate(f, dim=1, N=101, integration_domain=[[-xi + 1e-7, torch.pi / 2 - 1e-7]])

                return alpha * intg / torch.pi / torch.abs(torch.tensor(alpha - 1)) / (x0 - zeta)

            else:
                return self.pdf(-x, alpha, -beta)
        else:
            raise NotImplementedError("This function doesn't handle when alpha==1")

    def _pdf_simple(self, x: torch.Tensor, alpha):
        """
            simplified version of func `pdf_zolotarev`,
            assume alpha > 1 and beta = 0
        """
        assert (alpha > 1)

        x = torch.abs(x)

        def V(theta):
            return (torch.cos(theta) / torch.sin(alpha * theta)) ** (alpha / (alpha - 1)) * \
                   (torch.cos((alpha - 1) * theta) / torch.cos(theta))

        def g(theta):
            return V(theta) * x ** (alpha / (alpha - 1))

        def f(theta):
            g_ret = g(theta)
            g_ret = torch.nan_to_num(g_ret, posinf=0, neginf=0)

            return g_ret * torch.exp(-g_ret)

        simp = Simpson()
        intg = simp.integrate(f, dim=1, N=999, integration_domain=[[1e-7, torch.pi / 2 - 1e-7]])

        ret = alpha * intg / torch.pi / torch.abs(torch.tensor(alpha - 1, dtype=torch.float32)) / x

        if torch.any(torch.abs(x) < 2e-2):
            gamma_func = lambda a: torch.exp(torch.special.gammaln(a))

            alpha = torch.tensor(alpha, dtype=torch.float32)

            ret[torch.abs(x) < 2e-2] = gamma_func(1 + 1 / alpha) / torch.pi

        return ret

    def _score_simple(self, x: torch.Tensor, alpha):
        x.requires_grad_()
        log_likelihood = torch.log(self._pdf_simple(x, alpha))
        grad = torch.autograd.grad(log_likelihood.sum(), x, allow_unused=True)[0]

        # code above doesn't well estimate the score when |x| < 0.05
        # so, do linear approximation when |x| < 0.05
        if torch.any(torch.abs(x) < 0.05):
            grad_0_05 = self._score_simple(torch.tensor(0.05), alpha)
            grad[torch.abs(x) < 0.05] = x[torch.abs(x) < 0.05] * 20 * grad_0_05

        return grad
    
    def gen_skewed_levy(self,
                        alpha, 
                        size, 
                        device = None, 
                        isotropic = True,
                        clamp_a = 2000):
        assert 0 < alpha <= 2.
        if alpha == 2.:
            ret = 2 * torch.ones(size)
        else:
            ret = self.sample(alpha / 2., 
                            beta = 1, 
                            size = size, 
                            loc = 0., 
                            scale = 1., 
                            type = torch.float32, 
                            clamp = clamp_a,
                            is_isotropic = False) # need is_isotropic == false because of the code in sample
        if device is not None:
            ret = ret.to(device)
        return ret

    def gen_sas(self,
                alpha, 
                size, 
                a = None, 
                device = None, 
                isotropic = True,
                clamp_eps = 20000):
        if a is not None:
            ret = torch.randn(size=size, device=device)
            ret = torch.clamp(torch.sqrt(a)* ret, -clamp_eps, clamp_eps)
        else:
            ret = self.sample(alpha, 
                            beta = 0., 
                            size = size, 
                            loc = 0., 
                            scale = 1., 
                            type = torch.float32, 
                            clamp_threshold = clamp_eps,
                            is_isotropic = isotropic)
        if device is not None:
            ret = ret.to(device)
        return ret

    def sample(self, alpha, beta=0, size=1, loc=0, scale=1, type=torch.float32, reject_threshold:int=None,
               is_isotropic=False, clamp_threshold:int=None, clamp:int=None):

        assert 0 < alpha <= 2

        if isinstance(size, int):
            size_scalar = size
            size = (size,)
        else:
            size_scalar = 1
            for i in size:
                size_scalar *= i

        if is_isotropic and 0 < alpha < 2:
            assert not isinstance(size, int)

            num_sample = size[0]
            dim = int(size_scalar / num_sample)

            x = self._sample(alpha / 2, beta=1, size=num_sample * 2, type=type)
            # front
            if clamp is not None:
                x = torch.clamp(x, -clamp, clamp)
                
            x = x * 2 * torch.cos(torch.tensor([torch.pi * alpha / 4], dtype=torch.float64)) ** (2 / alpha)
            x = x.reshape(-1, 1)
            
            # backward
            # if clamp is not None:
                # x = torch.clamp(x, -clamp, clamp)

            z = torch.randn(size=(num_sample * 2, dim))
            e = x ** (1 / 2) * z
            e = (e * scale) + loc
            
            if reject_threshold is not None:
                e = e[torch.norm(e, dim=1) < reject_threshold]
            if clamp_threshold is not None:
                indices = e.norm(dim=1) > clamp_threshold
                e[indices] = e[indices] / e[indices].norm(dim=1)[:, None] * clamp_threshold
            
            # Gaussian resampling + forward
            # indices = torch.abs(e) > 5
            # e[indices] = torch.randn_like(e[indices]) * math.sqrt(2)
            
            # print(f"e min : {torch.min(e)} / max : {torch.max(e)}")
            # print(f"e mean : {torch.mean(e)} / std : {torch.std(e)}")
            # print(f"e.shape : {e.shape}")
            # print("- - - - - - - - - - - -")
            a = e[:num_sample].reshape(size).to(type)

            return a

        else:
            e = self._sample(alpha, beta=beta, size=size_scalar * 2, type=type)
            e = (e * scale) + loc
            if reject_threshold is not None:
                e = e[torch.abs(e) < reject_threshold]
            return e[:size_scalar].reshape(size)

    def _sample(self, alpha, beta=0, size=1, type=torch.float32):

        def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return 2 / torch.pi * ((torch.pi / 2 + bTH) * tanTH
                                   - beta * torch.log((torch.pi / 2 * W * cosTH) / (torch.pi / 2 + bTH)))

        def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return (W / (cosTH / torch.tan(aTH) + torch.sin(TH)) *
                    ((torch.cos(aTH) + torch.sin(aTH) * tanTH) / W) ** (1 / alpha))

        def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            # alpha != 1 and beta != 0
            val0 = beta * torch.tan(torch.tensor([torch.pi * alpha / 2], dtype=torch.float64))
            th0 = torch.arctan(val0) / alpha
            val3 = W / (cosTH / torch.tan(alpha * (th0 + TH)) + torch.sin(TH))
            res3 = val3 * ((torch.cos(aTH) + torch.sin(aTH) * tanTH -
                            val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W) ** (1 / alpha)
            return res3

        # TH = torch.rand(size, dtype=torch.float64) * torch.pi - (torch.pi / 2.0)
        TH = torch.rand(size, dtype=torch.float64) * (torch.pi - 0.3) - ((torch.pi - 0.3) / 2.0)
        W = Exponential(torch.tensor([1.0])).sample([size]).reshape(-1)
        aTH = alpha * TH
        bTH = beta * TH
        cosTH = torch.cos(TH)
        tanTH = torch.tan(TH)

        if alpha == 1:
            return alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        elif beta == 0:
            return beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        else:
            return otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)


@lru_cache(maxsize=1050)
def _get_pdf_dict(alpha):
    levy = LevyStable_is()
    x = torch.arange(-10, 10, 0.01)
    pdf = levy.pdf(x, alpha)
    dense_dict = TorchDictionary(keys=x, values=pdf)

    x = torch.arange(-100, 100, 0.1)
    pdf = levy.pdf(x, alpha)
    large_dict = TorchDictionary(keys=x, values=pdf)

    return dense_dict, large_dict
