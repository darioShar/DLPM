from tqdm import tqdm
import torch
import torch.nn as nn
import torch as th
from .LIM.functions.sde import VPSDE
from .LIM.functions.sampler import LIM_sampler
from .LIM.torchlevy import LevyStable
from .LIM.functions.loss import loss_fn
from .dlpm import DLPM, ModelMeanType, ModelVarType, LossType
from .dlpm import match_last_dims

# return 1-d vector (num batches) of loss
def compute_loss(tens, lploss):
    return torch.pow(torch.linalg.norm(tens, 
                                        ord = lploss, # always 2. here instead of lploss 
                                        dim = list(range(1, len(tens.shape)))), 
                                        1 / lploss)

def compute_loss_terms(x, y, lploss):
    if lploss == 2.: # L2 loss
        tmp = nn.functional.mse_loss(x, y, reduction='none')
        tmp = torch.sqrt(tmp.mean(dim = list(range(1, len(x.shape)))))
        return tmp
    elif lploss == 1.: # L1 loss
        tmp = nn.functional.smooth_l1_loss(x, y, beta=1, reduction='none')
        tmp = tmp.mean(dim = list(range(1, len(x.shape))))
        return tmp
    elif lploss == -1: # squared L2 loss
        return nn.functional.mse_loss(x, y, reduction='none').mean(dim = list(range(1, len(x.shape))))
    else:
        return compute_loss(x - y, lploss)



class GenerativeLevyProcess:
    """
    Utilities for training and sampling DLPMa and LIM diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/openai/improved-diffusion

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs (only epsilon for now).
    :param model_var_type: a ModelVarType determining how variance is output (only fixed variance for now).
    :param rescale_timesteps: if True, pass floating point timesteps into the model so that they are always scaled in (0 to 1).
    """

    def __init__(
            self, 
            alpha,
            device,
            reverse_steps,
            model_mean_type = ModelMeanType.EPSILON,
            model_var_type = ModelVarType.FIXED,
            time_spacing = 'linear',
            rescale_timesteps=False,
            isotropic = True, # isotropic levy noise
            LIM = False, # set dlpm heavy-tailed diffusion to continuous time LIM
            scale = 'scale_preserving',
            input_scaling = False,
            ):
        
        self.alpha = alpha
        self.device = device
        self.reverse_steps = reverse_steps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.time_spacing = time_spacing
        self.rescale_timesteps = rescale_timesteps
        self.isotropic = isotropic
        self.LIM = LIM
        self.input_scaling = input_scaling

        assert (self.model_mean_type == ModelMeanType.EPSILON) \
            and (self.model_var_type == ModelVarType.FIXED), \
            'Only epsilon prediction and fixed variance are supported for the moment'
        
        if self.LIM:
            assert (self.model_mean_type == ModelMeanType.EPSILON) \
            and (self.model_var_type == ModelVarType.FIXED) \
            and (self.rescale_timesteps == True), "LIM only supports epsilon prediction, fixed variance and rescaled timesteps"
            self.sde = VPSDE(alpha, 'cosine')
            self.levy = LevyStable()
        
        self.dlpm = DLPM(alpha,
                        device,
                        diffusion_steps=reverse_steps,
                        time_spacing = time_spacing,
                        isotropic = isotropic,
                        scale=scale)
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # return t.float() * (1.0 / self.reverse_steps)
            return t.float() * (1.0 / self.reverse_steps)
        return t

    def get_timesteps(self, N, **kwargs):
        return self.dlpm.get_timesteps(N)

    #####################################################################################
    ''' SAMPLING'''
    #####################################################################################

    def q_sample(self, x_start, t, eps = None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start, and eps
        """
        return self.dlpm.sample_x_t_from_xstart(x_start, t, eps)

    def q_posterior_mean_variance(self, x_start, x_t, t, Sigmas):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0, a_{1:T})
            we give Sigmas rather than a_{1:T}, since we want to compute the Sigmas only once.
        """
        assert x_start.shape == x_t.shape
        eps = self.dlpm.predict_eps(x_t, t, x_start)
        posterior_mean = self.dlpm.anterior_mean_dlpm(x_t, t, eps, Sigmas)
        posterior_variance = self.dlpm.anterior_variance_dlpm(t, Sigmas)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance

    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.
    :param model: the model, which takes a signal and a batch of timesteps
                    as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
                        x_start prediction before it is used to sample. Applies before
                        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
                        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
                - 'mean': the model mean output.
                - 'gamma': the model gamma output, if learned
                - 'Sigma': the forward variance. The reverse variance is Sigma_tilde_t= Gamma_t*Sigma_{t-1}
                - 'xstart': the prediction for x_0.
    """
    def p_mean_variance(self, 
                        model, 
                        x, 
                        t,
                        clip_denoised=False, 
                        denoised_fn=None, 
                        model_kwargs=None):
        # to process outpout of model
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        # get constants
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # run model
        input_scaling = 1.0
        if self.input_scaling and (self.dlpm.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1+self.dlpm.barsigmas[t]), x.shape)
        model_output = model(x * input_scaling,  self._scale_timesteps(t), **model_kwargs)

        if (self.model_mean_type == ModelMeanType.EPSILON) and (not clip_denoised):
            # just bypass everyhting
            model_eps = model_output
        else:
            if self.model_mean_type == ModelMeanType.EPSILON:
                model_xstart = process_xstart(
                    self.dlpm.predict_xstart(x_t=x, t=t, eps=model_output)
                )
            elif self.model_mean_type == ModelMeanType.START_X:
                model_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.Z:
                eps = torch.sqrt(self.dlpm.A[t]) * model_output
                model_xstart = process_xstart(
                    self.dlpm.predict_xstart(x_t=x, t=t, eps=eps)
                )
            elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                # get xstart from x_previous. use model_gamma
                eps = self.dlpm.predict_eps_from_m_tilde(x, t, model_output)
                model_xstart = process_xstart(
                    self.dlpm.predict_xstart(x_t=x, t=t, eps=eps)
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            # now get the new values after potential clipping and denoising
            model_eps = self.dlpm.predict_eps(x_t=x, t=t, xstart=model_xstart)

        # model variance is fixed for the moment, equal to Sigma_tilde_{t-1} = Gamma_t * Sigma_{t-1}
        model_mean, model_variance = self.dlpm.anterior_mean_variance_dlpm(x, t[0], model_eps)

        # some assertion
        assert model_mean.shape == x.shape, 'model mean:{}, x:{}'.format(model_mean.shape, x.shape)
        
        return {
            'eps': model_eps,
            'mean': model_mean,
            'variance': model_variance,
        }

    
    """
    Sample x_{t-1} from the model at the given timestep.
    """
    def p_sample(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = ((t != 1).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 1
        sample = out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise
        return {"sample": sample}#, "xstart": out["xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        progress=False,
        get_sample_history = False
    ):
        """
        Generate samples from the model.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """

        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total = self.reverse_steps)
        model.eval()
        x_hist = []

        with th.inference_mode():
            final = None
            for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                
                if progress:
                    pbar.update(1)
            
            if progress:
                pbar.close()
                tqdm._instances.clear()

            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Generate samples from the model and yield intermediate samples from each timestep 
        of diffusion. Returns a generator over dicts, where each dict is the return value of p_sample().
        """
        assert self.device is not None 
        assert isinstance(shape, (tuple, list))

        self.dlpm.sample_A(shape, self.reverse_steps)
        self.dlpm.compute_Sigmas()

        if noise is not None:
            img = noise
        else:
            img = self.dlpm.barsigmas[-1] * self.dlpm.gen_eps.generate(size = shape)
        yield {'sample': img}

        # linear timesteps
        indices = list(range(self.reverse_steps-1, 0, -1))

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            yield out
            img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = out['eps']
        model_mean, model_variance = self.dlpm.anterior_mean_variance_dlim(x, t, eps, eta=eta)
        assert not torch.isnan(model_mean).any() # verify no nan
        # deterministic sampling
        if eta == 0.0:
            return {"sample": model_mean}

        sample = model_mean + torch.sqrt(model_variance)*torch.randn_like(x)
        return {"sample": sample}#, "xstart": out["xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        progress=False,
        eta=0.0,
        get_sample_history = False
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """

        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total = self.reverse_steps)

        model.eval()
        x_hist = []
        with th.inference_mode():
            final = None
            for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)
            if progress:
                pbar.close()
                tqdm._instances.clear()
            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        assert self.device is not None
        assert isinstance(shape, (tuple, list))

        self.dlpm.sample_A(shape, self.reverse_steps)
        self.dlpm.compute_Sigmas()

        if noise is not None:
            img = noise
        else:
            img = self.dlpm.barsigmas[-1] * self.dlpm.gen_eps.generate(size = shape)
        yield {'sample': img}
        indices = list(range(self.reverse_steps-1, 0, -1))
        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            out = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]

    def lim_sample(self,
                   model,
                   shape,
                   ddim = False,
                   get_sample_history = False,
                   clip_denoised = False,
                   ):
                    
        init_clamp = 20. #config.sampling.init_clamp # is 20 in LIM configs, can be set to None
        
        x = self.dlpm.gen_eps.generate(size = shape)

        # if self.sde.alpha == 2.0:
        #     # Gaussian noise
        #     x = torch.randn(shape).to(self.device)
        # else:
        #     if self.isotropic:
        #         # isotropic
        #         x = self.levy.sample(alpha=self.sde.alpha, size=shape, is_isotropic=True, clamp=init_clamp).to(self.device)
        #     else:
        #         # non-isotropic
        #         x = torch.clamp(self.levy.sample(self.sde.alpha, size=shape, is_isotropic=False, clamp=None).to(self.device), 
        #                         min=-init_clamp, max=init_clamp)
            
        if False: #config.model.is_conditional:
            if config.sampling.cond_class is not None: 
                y = torch.ones(n) * config.sampling.cond_class
                y = y.to(self.device)
                y = torch.tensor(y, dtype=torch.int64)
            else:
                y = None
        else:
            y = None
            
        #x = LIM_sampler(args, config, x, y, model, self.sde, self.levy)
        samples = LIM_sampler(#args=self.config, 
                              #config=self.config,
                              ddim = ddim,
                              x = x,
                              y = y,
                              model = model,
                              sde = self.sde,
                              levy = self.levy,
                              isotropic=self.isotropic,
                              steps=self.reverse_steps,
                              gen_a = self.dlpm.gen_a,
                              gen_eps = self.dlpm.gen_eps,
                              device = self.device,
                              get_sample_history = get_sample_history)
        # The clamping and inverse affine transform will be managed in the generation manager.
        #x = x.clamp(-1.0, 1.0)
        #x = (x + 1) / 2
        return samples
    
    #####################################################################################
    ''' BEM: SAMPLING '''
    #####################################################################################

    def sample(self, 
                models,
                shape,
                reverse_steps,
                time_spacing = None,
                initial_data = None, 
                clip_denoised=False, 
                deterministic=False, 
                dlim_eta=1.0, 
                print_progression=False, 
                get_sample_history=False,
                clamp_a = None,
                clamp_eps = None):
        
        self.dlpm.gen_a.setParams(clamp_a = clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps = clamp_eps)

        model = models['default']

        # rescale noising with the number of provided reverse_steps
        assert time_spacing is None, "Specific time spacing is not yet supported for diffusion reverse sampling"
        
        default_reverse_steps = self.reverse_steps # store original number to restore.
        default_time_spacing = self.time_spacing
        if self.reverse_steps != reverse_steps:
            # if training reverse steps differs from evaluation reverse steps
            assert self.rescale_timesteps, "Rescaling only works when rescale_timesteps is True" 
            self.dlpm.rescale_diffusion(reverse_steps, time_spacing=time_spacing)
            self.reverse_steps = reverse_steps

        if self.LIM:
            x = self.lim_sample(model, 
                                shape = shape,
                                ddim = deterministic,
                                get_sample_history = get_sample_history,
                                clip_denoised = clip_denoised)
        else:
            if deterministic:
                x = self.ddim_sample_loop(model,
                                        shape = initial_data.shape if initial_data is not None else shape,
                                        noise = initial_data if initial_data is not None else None,
                                        eta = dlim_eta,
                                        progress=print_progression,
                                        get_sample_history = get_sample_history,
                                        clip_denoised = clip_denoised)
            else:
                x = self.p_sample_loop(model,
                                        shape = shape,
                                        progress=print_progression,
                                        get_sample_history = get_sample_history,
                                        clip_denoised = clip_denoised)
        
        # restore original diffusion steps
        if self.reverse_steps != reverse_steps:
            self.dlpm.rescale_diffusion(default_reverse_steps, default_time_spacing)
            self.reverse_steps = default_reverse_steps
        
        return x





    #####################################################################################
    ''' TRAINING '''
    #####################################################################################
    


    def training_losses(self, 
                        models, 
                        x_start, 
                        model_kwargs=None,
                        **kwargs):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, a dict of the noises at that step.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        model = models['default']
        x_start = x_start.to(self.device)
        # model_kwargs for model specific arguments, mainly conditioning
        if model_kwargs is None:
            model_kwargs = {}

        if self.LIM:
            loss = self.training_losses_lim(model, x_start, **model_kwargs, **kwargs)
        else:
            loss = self.training_losses_dlpm(model, x_start, **model_kwargs, **kwargs)
        return {'loss': loss}
 
    # For the moment, we only provide the implementation of Proposition (9) loss, which is the simple loss
    def training_losses_dlpm(self,
                            model,
                            x_start,
                            loss_type = 'EPSILON',
                            lploss = 2.0,
                            loss_monte_carlo  = 'mean',
                            monte_carlo_outer = 1,
                            monte_carlo_inner = 1,
                            model_kwargs = None,
                            clamp_a = None,
                            clamp_eps = None,
                            ):
        assert self.model_mean_type == ModelMeanType.EPSILON, 'only epsilon model output is supported for the moment'
        assert loss_type == LossType.EPS_LOSS, 'only epsilon loss is supported for the moment'
        if model_kwargs is None:
            model_kwargs = {}

        self.dlpm.gen_a.setParams(clamp_a = clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps = clamp_eps)

        
        # get timesteps
        t = torch.randint(1, self.reverse_steps, size=[len(x_start)]).to(self.device)
        # if self.dlpm.scale == 'scale_preserving':
        #     t = torch.randint(1, self.reverse_steps, size=[len(x_start)]).to(self.device)
        # elif self.dlpm.scale == 'scale_exploding':
        #     P_mean = -1.2
        #     P_std = 1.2
        #     t = torch.exp(torch.normal(P_mean, P_std, size=[len(x_start)]).to(self.device)).int()
        # else:
        #     raise NotImplementedError(self.dlpm.scale)
        
        # setup median of means estimator
        total_monte_carlo = monte_carlo_outer*monte_carlo_inner
        x_start_extended = x_start.repeat(total_monte_carlo, *([1]*len(x_start.shape[1:])))
        t_extended = t.repeat(total_monte_carlo)
        outer_shape = torch.tensor(x_start.shape) 
        outer_shape[0] *= monte_carlo_outer 
        A = self.dlpm.get_one_rv_faster_sampling(list(outer_shape))
        A_extended = A.repeat(monte_carlo_inner, *([1]*len(A.shape[1:])))
        z_t_extended = torch.randn_like(x_start_extended, device=self.device) # inner expectation Gaussian (z_t)
        # print('shapes', x_start_extended.shape, t_extended.shape, A_extended.shape, z_t_extended.shape)
        # get loss elements
        x_t, eps_t = self.dlpm.get_one_rv_loss_elements(t_extended, x_start_extended, A_extended, z_t_extended)

        # run model
        input_scaling = 1.0
        if self.input_scaling and (self.dlpm.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1+self.dlpm.barsigmas[t_extended]), x_t.shape)
        model_eps = model(x_t * input_scaling,  self._scale_timesteps(t_extended), **model_kwargs)
        
        # assert model_eps.shape == x_start_extended.shape

        # compute loss with the right exponent
        losses = compute_loss_terms(model_eps, eps_t, lploss)
        assert not torch.isnan(losses).any(), 'Nan in losses'

        if loss_monte_carlo == 'mean':
            loss = losses.mean()#(dim = 0)
        elif loss_monte_carlo == 'median':
            # Run median of means (rather than mean of medians)
            losses = losses.reshape(monte_carlo_outer, monte_carlo_inner, x_start.shape[0])
            losses = losses.mean(dim = 1)
            losses, _ = losses.median(dim = 0)
            loss = losses.mean()
        return loss
        

    def training_losses_lim(self, 
                            model, 
                            x_start, 
                            y = None,
                            clamp_a = None,
                            clamp_eps = None,
                            ):
        n = x_start.size(0)
        self.dlpm.gen_a.setParams(clamp_a = clamp_a)
        self.dlpm.gen_eps.setParams(clamp_eps = clamp_eps)
        # Noise
        if self.sde.alpha == 2.0:
            # gaussian noise
            e = torch.randn_like(x_start).to(self.device)
        else:
            clamp = 20 # set to 20 in their configs. Can be set to None
            e = self.dlpm.gen_eps.generate(size = x_start.shape)
            # if self.isotropic:
            #     e = self.levy.sample(alpha=self.sde.alpha, size=x_start.shape, is_isotropic=True, clamp=clamp).to(self.device)
            # else:
            #     e = torch.clamp(self.levy.sample(self.sde.alpha, size=x_start.shape, is_isotropic=False, clamp=None).to(self.device), 
            #                     min=-clamp, max=clamp)

        # time
        start_eps = 1e-5  
        t = torch.rand(n).to(self.device) * (self.sde.T - start_eps) + start_eps        
        losses = loss_fn(model, self.sde, x_start, t, e, config = None, y = None)
        assert not torch.isnan(losses).any(), 'Nan in losses'
        # loss = losses.mean()
        return losses

