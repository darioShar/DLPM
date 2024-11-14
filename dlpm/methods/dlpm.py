import torch
import bem.datasets.Data as Data
from bem.datasets.Distributions import *




### TO REIMPLEMENT
### For the moment, only implementing 'EPSILON' prediction, 'FIXED' variance, 'EPS_LOSS'
class ModelMeanType():
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = 'PREVIOUS_X'  # the model predicts x_{t-1}
    START_X = 'START_X'  # the model predicts x_0
    EPSILON = 'EPSILON'  # the model predicts epsilon
    Z = 'Z' # the model predicts z_t
    SQRT_GAMMA_EPSILON = 'SQRT_GAMMA_EPSILON' # the model predicts (1 - sqrt(1 - gamma))eps

class ModelVarType():
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED = 'FIXED' # fixed variance
    # GAMMA = 'GAMMA' # can learn gamma factor, with the true var being tilde_sigma = gamma * sigma
    # SIGMA_T_1_OVER_A = enum.auto() # learn directly Sigma_t_1 / a_t
    # TRUE_VAR = enum.auto()

class LossType():
    # can replace different means by normal loss and lambda loss, with some lambda to
    # pass as argument in training. should be better.
    LP_LOSS = 'LP_LOSS' # L_p loss between model output and target value
    MEAN_LOSS = 'MEAN_LOSS' # LP loss between model mean and true mean
    EPS_LOSS = 'EPS_LOSS' # lp loss between noise and model noise
    LAMBDA_LOSS = 'LAMBDA_LOSS' #  L_p loss between model output and added noise, times lambda factor (=eps * lambda factor)
    VAR_KL = 'VAR_KL' # when mixing mean and var.
    VAR_LP_SUM = 'VAR_LP_SUM' # when mixing mean and var.


# repeat a tensor so that its last dimensions [1:] match size[1:]
# ideal for working with batches.
def match_last_dims(data, size):
    assert len(data.size()) == 1 # 1-dimensional, one for each batch
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))


# Compute the right noise schedules, with arbitrary gamma and sigma, so no more bg (bargammas)

class DLPM:
    def __init__(
        self,
        alpha,
        device,
        diffusion_steps,
        time_spacing = 'linear',
        isotropic = True, # isotropic levy noise
        clamp_a = None,
        clamp_eps = None,
        scale = 'scale_preserving'
    ):
        self.alpha = alpha
        self.device = device
        self.time_spacing = time_spacing
        self.isotropic = isotropic
        self.use_single_a_chain = True
        self.scale = scale

        # 1d noising schedules
        self.gammas, self.bargammas, self.sigmas, self.barsigmas = \
                    (x.to(self.device) for x in self.gen_noise_schedule(diffusion_steps, scale=self.scale))
        
        # noising schedules, adapted to batch size, to be potentially updated for each incoming batch
        self.constants = None

        # self.bargammas = torch.cumprod(self.gammas, dim = 0)
        self.gen_a = Data.Generator('skewed_levy', 
                                    alpha = self.alpha, 
                                    device=self.device,
                                    isotropic=isotropic,
                                    clamp_a = clamp_a)
        
        self.gen_eps = Data.Generator('sas',
                                      alpha = self.alpha, 
                                      device=self.device,
                                      isotropic=isotropic,
                                      clamp_eps = clamp_eps)
        
        self.A = None
        self.Sigmas = None


#######################################################################################
    ''' Noise schedules '''
#######################################################################################
    
    def get_timesteps(self, steps):
        if self.time_spacing == 'linear':
            timesteps = torch.tensor(range(0, steps), dtype=torch.float32)
        elif self.time_spacing == 'quadratic':
            timesteps = steps* (torch.tensor(range(0, steps), dtype=torch.float32) / steps)**2
        else:
            raise NotImplementedError(self.time_spacing)
        return timesteps


    # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
    def gen_noise_schedule(self, diffusion_steps, scale = 'scale_preserving'):
        # assert scale == 'scale_preserving', 'Only scale-preserving schedule is implemented for now'
        if scale == 'scale_preserving':
            s = 0.008
            timesteps = self.get_timesteps(diffusion_steps)

            schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

            baralphas = schedule / schedule[0]
            betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
            alphas = 1 - betas

            # linear schedule for gamma
            gammas = alphas**(1/self.alpha)
            bargammas = torch.cumprod(gammas, dim = 0)

            # scale-preserving schedule
            sigmas = (1 - gammas**(self.alpha))**(1/self.alpha)
            barsigmas = (1 - bargammas**(self.alpha))**(1/self.alpha)
        
        elif scale == 'scale_exploding':
            timesteps = self.get_timesteps(diffusion_steps)
            sigma_min = 0.002
            sigma_max = 80
            rho = 7

            gammas = torch.ones_like(timesteps)
            bargammas = torch.ones_like(timesteps)

            barsigmas = (sigma_min**(1 / rho) + (timesteps / (diffusion_steps - 1)) * (sigma_max**(1 / rho) - sigma_min**(1 / rho)))**rho
            # Calculate sigmas based on the cumulative sum condition
            barsigmas_alpha = barsigmas**self.alpha
            sigmas_alpha = torch.ones_like(barsigmas) * barsigmas_alpha[0]
            for i in range(1, len(barsigmas)):
                sigmas_alpha[i] = barsigmas_alpha[i] - torch.sum(sigmas_alpha[:i])
            sigmas = sigmas_alpha**(1 / self.alpha)
            
            # assert torch.all(sigmas > 0), 'Sigma must be positive'
            
        else:
            assert False, 'Unknown scale'
        
        return gammas, bargammas, sigmas, barsigmas

    def get_schedule(self, shape):
        g = match_last_dims(self.gammas, shape)
        bg = match_last_dims(self.bargammas, shape)
        s = match_last_dims(self.sigmas, shape)
        bs = match_last_dims(self.barsigmas, shape)
        return g, bg, s, bs
    
    def get_t_to_batch_size(self, x_t, t):
        if isinstance(t, int):
            return torch.full([x_t.shape[0]], t).to(self.device)
        return t
    
    # in order not to compute the gammas, bargammas constants each time.
    def update_constants(self, shape):
        if (self.constants is None) or (self.constants[0].shape != shape):
            self.constants = self.get_schedule(shape)
        return self.constants
    
    def rescale_diffusion(self, diffusion_steps, time_spacing = None):
        assert isinstance(diffusion_steps, int), "Diffusion steps must be an integer"
        # just update with this number of diffusion steps regenerate noise schedule
        if time_spacing is not None:
            self.time_spacing = time_spacing
        # 1d noising schedules
        self.gammas, self.bargammas, self.sigmas, self.barsigmas = \
                    (x.to(self.device) for x in self.gen_noise_schedule(diffusion_steps))
        # noising schedules, adapted to batch size, to be potentially updated for each incoming batch
        self.constants = None

#######################################################################################
    ''' Dynamics with no data augmentation/conditioning on A_{1:T} '''
#######################################################################################
    
    def predict_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        xstart = (x_t - eps*bs[t]) / bg[t]
        return xstart 

    def predict_eps(self, x_t, t, xstart):
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        eps = (x_t - xstart* bg[t]) / bs[t]
        return eps
    
    def predict_eps_from_m_tilde(self, x_t, t, m_tilde_t_1):
        g, bg, s, bs = self.update_constants(m_tilde_t_1.shape)
        t = self.get_t_to_batch_size(m_tilde_t_1, t)
        Gamma_t = self.compute_Gamma_t(t, self.Sigmas[t-1], self.Sigmas[t])
        eps = (x_t - m_tilde_t_1*g[t]) / (bs[t]*Gamma_t) # Since m_tilde_t_1 = (x_t - bs[t]*Gamma_t*eps) / g[t]
        return eps
    
    def sample_x_t_from_xstart(self, xstart, t, eps=None):
        g, bg, s, bs = self.update_constants(xstart.shape)
        t = self.get_t_to_batch_size(xstart, t)
        if eps is None:
            eps = self.gen_eps.generate(size = xstart.size())
        x_t = bg[t]*xstart + bs[t]*eps
        return x_t, eps
    
    
#######################################################################################
    ''' Dynamics with data augmentation/conditioning on A_{1:T} '''
#######################################################################################

    ''' We will use a whole sequence (A_t), (Sigma_t) only for sampling'''
    # Get the sequence A_{1:T} to use for the a full sampling path
    def sample_A(self, shape, diffusion_steps):
        self.A = torch.stack([self.gen_a.generate(size = shape) for i in range(diffusion_steps)])

    # Compute Sigma_t from the list A
    def compute_Sigmas(self):
        g, bg, s, bs = self.update_constants(self.A[0].shape)
        # Sigmas[t] = s[t]^2 a_t + g[t]^2 Sigmas[t-1]
        t = self.get_t_to_batch_size(self.A[0], 0)
        Sigmas = [s[0]**2 * self.A[0]]
        for t in range(1, self.A.shape[0]):
            A_t = self.A[t]
            t = self.get_t_to_batch_size(self.A[0], t)
            Sigmas.append(s[t]**2 * A_t + g[t]**2 * Sigmas[-1])
        self.Sigmas = torch.stack(Sigmas)
        

    def sample_x_t_from_xstart_given_Sigma(self, xstart, t, Sigma_t, z_t = None):
        g, bg, s, bs = self.update_constants(xstart.shape)
        t = self.get_t_to_batch_size(xstart, t)
        if z_t is None:
            z_t = torch.randn_like(xstart)
        x_t = bg[t]*xstart + Sigma_t**(1/2)*z_t
        return x_t
    
    def compute_Gamma_t(self, t, Sigma_t_1, Sigma_t):
        g, bg, s, bs = self.update_constants(Sigma_t_1.shape)
        t = self.get_t_to_batch_size(Sigma_t_1, t)
        Gamma_t = 1 - (g[t]**2* Sigma_t_1) / Sigma_t
        return Gamma_t
    
    def compute_Sigma_tilde_t_1(self, Gamma_t, Sigma_t_1):
        return Gamma_t * Sigma_t_1
    
    def compute_m_tilde_t_1(self, x_t, t, Gamma_t, eps_t):
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        m_tilde_t_1 = (x_t - bs[t]*Gamma_t*eps_t) / g[t]
        return m_tilde_t_1
    
#######################################################################################
    ''' Sampling '''
#######################################################################################
    '''
        Get x_{t-1} from x_t, given eps = eps_{theta}(x_t), and Sigmas(A_{1:T})
    '''

    def anterior_mean_variance_dlpm(self, x_t, t, eps):
        g, bg, s, bs = self.update_constants(x_t.shape)
        Gamma_t = self.compute_Gamma_t(t, self.Sigmas[t-1], self.Sigmas[t])
        t = self.get_t_to_batch_size(x_t, t)
        x_t_1 = (x_t - bs[t]*Gamma_t*eps) / g[t]
        Sigma_t_1 = self.compute_Sigma_tilde_t_1(Gamma_t, self.Sigmas[t-1])
        return x_t_1, Sigma_t_1


    def anterior_mean_variance_dlim(self, x_t, t, eps, eta = 0.0):
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        nonzero_mask = (t != 1).float().view(-1, *([1] * (len(x_t.shape) - 1))) # no noise when t == 1
        if eta == 0.0:
            sample = (x_t - bs[t]*eps) / g[t] + bs[t-1]*eps
            return sample, 0
        
        sigma_t = eta * bs[t-1]

        sample = (x_t - bs[t]*eps) / g[t]
        sample += (bs[t-1]**(self.alpha) - sigma_t**(self.alpha))**(1 / self.alpha)*eps
        mean = sample

        variance = nonzero_mask * sigma_t**2 * self.A[t]

        return sample, variance




#######################################################################################
    ''' Training: we use Proposition (8), and only sample two heavy-tailed r.v a_t_0, a_t_1 '''
#######################################################################################

    # returns a_t_0, a_t_1 as in Propostion (8)
    # must specify 't' to manage the case t==1
    def get_two_rv_faster_sampling(self, Xbatch, t):
        a_t_0 = self.gen_a.generate(size = Xbatch.size())
        # zero variance when t == 1
        zero_loc = torch.where(t == 1)[0]
        if zero_loc is not tuple():
            a_t_0[zero_loc] = torch.zeros(
                                (zero_loc.shape[0], 
                                    *(a_t_0.shape[1:]))
                                    ).to(self.device)
        a_t_1 = self.gen_a.generate(size = Xbatch.size())
        return a_t_0, a_t_1
    
    # As in Proposition (8)
    # Computes Sigma_prime_t_1 from a_t_0, s.t x_{t-1} =d bg[t-1] x_{t-2} + sigma_prime_{t-1}**(1/2) G_{t-1}
    # Compute Sigma_prime_t such that x_t =d bg[t] x_{t-1} + sigma_t**(1/2) G_t
    # Computes Gamma_prime_t = 1 - (g[t]**(2 / alpha) * Sigma_t_1) / Sigma_t
    def compute_two_rv_prime_constants(self, t, a_t_0, a_t_1):
        g, bg, s, bs = self.update_constants(a_t_0.shape)
        # Sigmas[t-1]
        Sigma_prime_t_1 = bs[t-1]**2 * a_t_0
        # Sigmas[t]. Same recurrence as in compute_Sigmas
        Sigma_prime_t = s[t]**2 * a_t_1 + g[t]**2 * Sigma_prime_t_1
        # Gammas[t]
        Gamma_prime_t = self.compute_Gamma_t(t, Sigma_prime_t_1, Sigma_prime_t)
        return Sigma_prime_t_1, Sigma_prime_t, Gamma_prime_t
    

    # It is very important to remember that here, (a_t_0, a_t_1, x_0, x_t) are part of the same sampled sequence
    # they are not independent, x_t is sampled as bg[t] x_0 + Sigma_prime_t**(1/2) G_t
    def compute_two_rv_tilde_constants(self, t, Sigma_prime_t_1, Gamma_prime_t, x_0, x_t):
        g, bg, s, bs = self.update_constants(x_t.shape)
        # Sigmas_tilde[t-1]
        Sigma_tilde_t_1 = Gamma_prime_t* Sigma_prime_t_1
        # predicted noise eps_t
        eps_t = self.predict_eps(x_t, t, x_0)
        # anterior mean m_tilde[t-1]
        m_tilde_t_1 = (x_t - bs[t]*Gamma_prime_t*eps_t) / g[t]

        return eps_t, m_tilde_t_1, Sigma_tilde_t_1
    
    # lambda, as presented in design choice D3. One can play with lambda and set it as a smarter function of t
    def compute_two_rv_lambda(self, t, Sigma_prime_t_1):
        g, bg, s, bs = self.update_constants(Sigma_prime_t_1.shape)
        t = self.get_t_to_batch_size(Sigma_prime_t_1, t)
        lambda_t = bs[t] / (2* g[t]* Sigma_prime_t_1)
        return lambda_t
    
    # now do the whole computation from t, x_0 
    def get_two_rv_loss_elements(self, t, x_0):
        # Sample the two r.v a_t_0, a_t_1
        a_t_0, a_t_1 = self.get_two_rv_faster_sampling(x_0, t)
        # Compute the equivalent Sigmas, Gamma
        Sigma_prime_t_1, Sigma_prime_t, Gamma_prime_t = self.compute_two_rv_prime_constants(t, a_t_0, a_t_1)
        # Sample x_t from x_0
        x_t = self.sample_x_t_from_xstart_given_Sigma(x_0, t, Sigma_prime_t)
        # we now have sampled (x_0, x_t, a_t_0, a_t_1)

        # Compute the elements relevant to the loss function
        eps_t = self.predict_eps(x_t, t, x_0) # added noise
        m_tilde_t_1 = self.compute_m_tilde_t_1(x_t, t, Gamma_prime_t, eps_t) # anterior mean
        Sigma_tilde_t_1 = self.compute_Sigma_tilde_t_1(Gamma_prime_t, Sigma_prime_t_1) # anterior variance
        lambda_t = self.compute_two_rv_lambda(t, Sigma_prime_t_1) # lambda, as presented in design choice D3
        return x_t, eps_t, m_tilde_t_1, Sigma_tilde_t_1, lambda_t

#######################################################################################
    ''' Training: we use Proposition (9), and only sample one heavy-tailed r.v a_t_0, a_t_1 '''
#######################################################################################
    
    ''' 
        Here we must satisfy design choices D1, D2, D3, so 
        we do not learn variance (set to Sigma_tilde)
        we do not compute lambda (set to 1)
        we only compute eps_t
    '''

    # returns a_t as in Propostion (9)
    def get_one_rv_faster_sampling(self, shape):
        return self.gen_a.generate(size = shape)

    # compute Sigma_t from a_t, as in Proposition (9)
    def compute_one_rv_Sigma_prime_t(self, t, a_t):
        g, bg, s, bs = self.update_constants(a_t.shape)
        t = self.get_t_to_batch_size(a_t, t)
        # only returns Sigmas[t]
        Sigma_prime_t = a_t * bs[t]**2 
        return Sigma_prime_t

    def get_one_rv_loss_elements(self, t, x_0, a_t = None, z_t = None):
        if a_t is None:
            a_t = self.get_one_rv_faster_sampling(x_0.shape)
        Sigma_prime_t = self.compute_one_rv_Sigma_prime_t(t, a_t)
        x_t = self.sample_x_t_from_xstart_given_Sigma(x_0, t, Sigma_prime_t, z_t=z_t)
        eps_t = self.predict_eps(x_t, t, x_0)
        return x_t, eps_t


    # def _eps_from_previous_x(self, t, xt, previous_x, Gamma, constants = None):
    #     g, bg = self.get_constants(constants, xt.size())
    #     return (xt - g[t]**(1/self.alpha)*previous_x) / (Gamma * (1 - bg[t])**(1/self.alpha))



    # def get_constants_to_incoming_batch_dims(self, Xbatch, t):
    #     g, bg, s, bs = self.get_schedule(Xbatch)
    #     t = self.get_t_to_batch_size(Xbatch, t)
    #     a_t_1, a_t_prime, a_t = self.get_noises_to_incoming_batch_dims(Xbatch, t)
    #     return g, bg, t, a_t_1, a_t_prime, a_t

