import torch
import math
import numpy as np

class VPSDE:
    def __init__(self, alpha, schedule='cosine', T=0.9946):
        self.beta_0 = 0
        self.beta_1 = 20
        self.alpha = alpha
        
        self.cosine_s = 0.008
        self.schedule = schedule
        self.cosine_beta_max = 0.999
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. \
                            * (1. + self.cosine_s) / math.pi - self.cosine_s
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = T

        else:
            self.T = 1.
         
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        
    
    def beta(self, t):
        if self.schedule =='linear':
            beta = (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi / 2 * self.alpha / (self.cosine_s + 1) * torch.tan((t + self.cosine_s)/(1 + self.cosine_s) * math.pi / 2)
        
        return beta

    def marginal_log_mean_coeff(self, t):
        if self.schedule =='linear':
            log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.pow(1. - torch.exp(self.marginal_log_mean_coeff(t)*self.alpha), 1 / self.alpha)

    def inverse_a(self, a):
        return 2/np.pi*(1+self.cosine_s)*torch.acos(a)-self.cosine_s