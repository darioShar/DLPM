import torch
import torch.nn.functional as F
import math

def match_last_dims(data, size):
    assert len(data.size()) == 1 # 1-dimensional, one for each batch
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))


def loss_fn(model, 
            sde,
            x0,
            t,
            e,
            config,
            y=None):

    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)
    

    x_t = x0 * match_last_dims(x_coeff, x0.shape) + e * match_last_dims(sigma, x0.shape)

    if sde.alpha == 2.0:
        score = - e  
    else:
        score = - e / sde.alpha

    if y is None:
        output = model(x_t, t)
    else:
        output = model(x_t, t, y)

    #d = x0.shape[-1] * x0.shape[-2] * x0.shape[-3]
    d = torch.prod(torch.tensor(x0.shape[1:]))

    out = F.smooth_l1_loss(output, score, beta=1, size_average=True, reduction='mean') # / d

    return out