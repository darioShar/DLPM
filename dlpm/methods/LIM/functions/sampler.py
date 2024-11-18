import torch
import torch.nn.functional as F
import math
import scipy
import numpy as np
import tqdm 

def match_last_dims(data, size):
    assert len(data.size()) == 1 # 1-dimensional, one for each batch
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))


def LIM_sampler(#args, 
                #config,
                ddim,
                x, 
                y, 
                model, 
                sde, 
                levy,
                isotropic,
                steps,
                gen_a,
                gen_eps,
                sde_clamp=None,
                masked_data=None, 
                mask=None, 
                t0=None, 
                device='cuda',
                get_sample_history=False,):
    
    #if args.sample_type not in ['sde', 'ode', 'sde_imputation']:
    #    raise Exception("Invalid sample type")
    
    # THEIR THEORY ONLY ALLOWS FOR ISOTROPY, SO LETS GO WITH ISOTROPY
    is_isotropic= isotropic #config.diffusion.is_isotropic 
    steps = steps #args.nfe
    eps = 1e-5
    method = 'ode' if ddim else 'sde' # args.sample_type
    
    # They are always using sde_clamp = 20
    # if sde_clamp is None:
    #     if clamp_a is None:
    #         sde_clamp = 20
    #     else:
    #         sde_clamp = clamp_a #config.sampling.sde_clamp
    
    # clamp_eps is what they call clamp_thresgold

    is_conditional = False #config.model.is_conditional

    def score_model(x, t):
        if is_conditional:
            out = model(x, t, y)
        else:
            out = model(x, t)
        return out

    def impainted_noise(data, noise, mask, t, device):
        sigma = sde.marginal_std(t)
        x_coeff = sde.diffusion_coeff(t)

        e_L = gen_eps.generate(size = data.shape)
        # if sde.alpha == 2:
        #     e_L = torch.randn(size=(data.shape)).to(device)
        # else:
        #     if is_isotropic:
        #         e_L = levy.sample(alpha=sde.alpha, size=data.shape, is_isotropic=True, clamp=sde_clamp,clamp_threshold = clamp_eps).to(device)
        #     else:
        #         e_L = torch.clamp(levy.sample(alpha=sde.alpha, size=data.shape, is_isotropic=False, clamp=None).to(device), 
                                    #   min=-sde_clamp, max=sde_clamp)
        
        match_last_dims(x_coeff, data.shape)
        data = match_last_dims(x_coeff, data.shape) * data + match_last_dims(sigma, data.shape) * e_L
        masked_data = data * mask + noise * (1-mask)

        return masked_data

    def ode_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        tmp = torch.pow(sde.marginal_std(s), -(sde.alpha-1))
        tmp = match_last_dims(tmp, x.shape)
        score_s = score_model(x, s) * tmp
        
        beta_step = sde.beta(s) * (s - t)

        x_coeff = sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1)
        
        if sde.alpha == 2:
            tmp = torch.pow(sde.marginal_std(s) + 1e-5, -(sde.alpha-1))
            tmp = match_last_dims(tmp, x.shape)
            score_s = score_model(x, s) * tmp

            x_coeff = 1 + beta_step/sde.alpha
            score_coeff = beta_step / 2

            x_t = match_last_dims(x_coeff, x.shape) * x + \
                match_last_dims(score_coeff, x.shape) * score_s
        
            #score_coeff = torch.pow(sde.marginal_std(s), sde.alpha-1) * sde.marginal_std(t) * ( 1- torch.exp(h_t))
            #x_t = match_last_dims(x_coeff, x.shape) * x + match_last_dims(score_coeff, x.shape) * score_s
        else:             
            a = sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s), -1)
            score_coeff = -sde.alpha*(1- a)
            
            x_t = match_last_dims(x_coeff, x.shape) * x + match_last_dims(score_coeff, x.shape) * score_s
            
        return x_t

    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        tmp = torch.pow(sde.marginal_std(s), -(sde.alpha-1))
        tmp = match_last_dims(tmp, x.shape)
        s_theta_tmp = score_model(x, s)
        score_s = s_theta_tmp * tmp
        beta_step = sde.beta(s) * (s - t)

        a = torch.exp(sde.marginal_log_mean_coeff(t) - sde.marginal_log_mean_coeff(s))
        x_coeff = a

        noise_coeff = torch.pow(-1 + torch.pow(a, sde.alpha), 1/sde.alpha)

        # To change?
        if sde.alpha == 2:
            tmp = torch.pow(sde.marginal_std(s) + 1e-5, -(sde.alpha-1))
            tmp = match_last_dims(tmp, x.shape)
            score_s = score_model(x, s) * tmp
            e_B = torch.randn_like(x).to(device)

            x_coeff = 1 + beta_step/sde.alpha
            score_coeff = beta_step
            noise_coeff = torch.pow(beta_step, 1 / sde.alpha)

            x_t = match_last_dims(x_coeff, x.shape) * x + \
                match_last_dims(score_coeff, x.shape) * score_s + \
                    match_last_dims(noise_coeff, x.shape) * e_B
        else:
            e_L = gen_eps.generate(size = x.shape)
            # if is_isotropic:
            #    e_L = levy.sample(alpha=sde.alpha, size=x.shape, is_isotropic=True, clamp=sde_clamp).to(device)
            # else:
            #     e_L = torch.clamp(levy.sample(alpha=sde.alpha, size=x.shape, is_isotropic=False, clamp=None).to(device), 
            #                       min=-sde_clamp, max=sde_clamp)
            score_coeff = sde.alpha ** 2 * (-1 + a)
            x_t = match_last_dims(x_coeff, x.shape) * x + \
                match_last_dims(score_coeff, x.shape) * score_s + \
                    match_last_dims(noise_coeff, x.shape) * e_L
        # end to change
        # ADDED HERE
        #e_L = get_eps.generate(size = x.shape)
        #score_coeff = sde.alpha ** 2 * (-1 + a)
        #x_t = match_last_dims(x_coeff, x.shape) * x + \
        #    match_last_dims(score_coeff, x.shape) * score_s + \
        #        match_last_dims(noise_coeff, x.shape) * e_L

        # clip denoised, our own addition too
        ### x_t = a_t x_0 + gamma_t e_{theta}, gamma_t = (1 - a**(alpha))**(1/alpha)
        ### - e_{theta} / (sde.alpha) = s_theta_tmp
        #if clip_denoised:
        #    a_t = torch.exp(sde.marginal_log_mean_coeff(t))
        #    #gamma_t = (1 - a_t**(sde.alpha))**(1/sde.alpha)
        #    s_t = sde.marginal_std(s)
        #    a_t = match_last_dims(a_t, x_t.shape)
        #    s_t = match_last_dims(s_t, x_t.shape)
        #    eps_theta = - s_theta_tmp * sde.alpha
        #    x_0 = (x_t - s_t * eps_theta) / a_t
        #    x_0 = x_0.clamp(-1., 1.)
        #    x_t = a_t * x_0 + s_t * eps_theta
        # END ADDED

        ### x_t = x_t.type(torch.float32)
       
        return x_t
    
    def sde_score_update_imputation(data, mask, x, s, t):
        score_s = score_model(x, s) * torch.pow(sde.marginal_std(s), -(sde.alpha-1))[:,None,None,None]

        beta_step = sde.beta(s) * (s - t)
        beta_step = (sde.marginal_log_mean_coeff(t)-sde.marginal_log_mean_coeff(s))*sde.alpha

        x_coeff = 1 + beta_step/sde.alpha
        a = sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s),-1)

        noise_coeff = torch.pow(-1+a,1/sde.alpha)

        if sde.alpha == 2:
            e_B = torch.randn_like(x).to(device)
           
            score_coeff = beta_step
            x_t = match_last_dims(x_coeff, x.shape) * x + match_last_dims(score_coeff, x.shape) * score_s + noise_coeff[:, None, None,None] * e_B
        else:
            e_L = gen_eps.generate(size = x.shape)
            # if is_isotropic:
            #     e_L = levy.sample(alpha=sde.alpha, size=x.shape, is_isotropic=True, clamp=sde_clamp,clamp_threshold = clamp_eps).to(device)
            # else:
            #     e_L = torch.clamp(levy.sample(alpha=sde.alpha, size=x.shape, is_isotropic=False, clamp=None).to(device), 
            #                       min=-sde_clamp, max=sde_clamp)
                
            score_coeff = sde.alpha * beta_step
            score_coeff =(sde.marginal_log_mean_coeff(t)-sde.marginal_log_mean_coeff(s))*sde.alpha**2
            score_coeff = sde.alpha**2*(-1+a)
            x_t = match_last_dims(x_coeff, x.shape) * x+ match_last_dims(score_coeff, x.shape) * score_s + noise_coeff[:, None, None,None] * e_L

        x_t = impainted_noise(data, x_t, mask, t,device)

        return x_t


    # Sampling steps    
    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)  # linear
    #timesteps = torch.pow(torch.linspace(np.sqrt(sde.T), np.sqrt(eps), steps + 1), 2).to(device) # quadratic
    
    if get_sample_history:
        history = [x]
    with torch.no_grad():
        if method == "sde_imputation":
            x = impainted_noise(masked_data, x, mask, t0, device)
            
        # for i in tqdm.tqdm(range(steps)):
        for i in range(steps):
            vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
            
            if method == 'sde':
                x = sde_score_update(x, vec_s, vec_t)
                # clamp threshold : re-normalization
                # if clamp_eps :
                #     size = x.shape
                #     l = len(x)
                #     x = x.reshape((l, -1))
                #     indices = x.norm(dim=1) > clamp_eps
                #     x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * clamp_eps
                #     x = x.reshape(size)
            
            elif method == 'sde_imputation':
                x = sde_score_update_imputation(masked_data, mask, x, vec_s, vec_t)
                x = sde_score_update(x, vec_s, vec_t)

                # size = x.shape
                # l = len(x)
                # x = x.reshape(( l, -1))
                # indices = x.norm(dim=1) > clamp_eps
                # x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * clamp_eps
                # x = x.reshape(size)
    
            elif method == 'ode':
                x = ode_score_update(x, vec_s, vec_t)
            if get_sample_history:
                history.append(x)
    if get_sample_history:
        return x, torch.stack(history)
    return x