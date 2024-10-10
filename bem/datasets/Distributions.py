# file to generate some default data like swiss roll, GMM, levy variables

from sklearn.datasets import make_swiss_roll
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import scipy

def match_last_dims(data, size):
    """
    Expands a 1-dimensional tensor so that its last dimensions match the target size.
    
    Args:
        data (torch.Tensor): A 1-dimensional tensor with shape [batch_size].
        size (tuple or list): The target size, where size[0] should match batch_size.
    
    Returns:
        torch.Tensor: The expanded tensor with shape `size`.
    """
    # Ensure data is 1-dimensional
    assert data.dim() == 1, f"Data must be 1-dimensional, got {data.size()}"
    
    # Unsqueeze to add singleton dimensions for expansion
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    
    # Use expand instead of repeat to save memory
    return data.expand(*size).contiguous()

''' Generate fat tail distributions'''
# assumes it is a batch size
# is isotropic, just generates a single 'a' tensored to the right shape
def gen_skewed_levy(alpha, 
                    size, 
                    device = None, 
                    isotropic = True,
                    clamp_a = None):
    if (alpha > 2.0 or alpha <= 0.):
        raise Exception('Wrong value of alpha ({}) for skewed levy r.v generation'.format(alpha))
    if alpha == 2.0:
        ret = 2 * torch.ones(size)
        return ret if device is None else ret.to(device)
    # generates the alplha/2, 1, 0, 2*np.cos(np.pi*alpha/4)**(2/alpha)
    if isotropic:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size[0]), dtype=torch.float32)
        ret = match_last_dims(ret, size)
    else:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size), dtype=torch.float32)
    if clamp_a is not None:
        ret = torch.clamp(ret, 0., clamp_a)
    return ret if device is None else ret.to(device)


#symmetric alpha stable noise of scale 1
# can generate from totally skewed noise if provided
# assumes it is a batch size
def gen_sas(alpha, 
            size, 
            a = None, 
            device = None, 
            isotropic = True,
            clamp_eps = None):
    if a is None:
        a = gen_skewed_levy(alpha, size, device = device, isotropic = isotropic)
    ret = torch.randn(size=size, device=device)
    
    #if device is not None:
    #    ret = ret.to(device)
    #ret = repeat_shape(ret, size)
    ret = torch.sqrt(a)* ret
    if clamp_eps is not None:
        ret = torch.clamp(ret, -clamp_eps, clamp_eps)
    return ret

    '''
        if isotropic:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size[0]), dtype=torch.float32)
            ret = repeat_shape(ret, size)
        else:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size), dtype=torch.float32)
        return ret if device is None else ret.to(device)
    else:
        if isotropic:
            ret = torch.randn(size=(size[0],))
            ret = repeat_shape(ret, size)
        else:
            ret = torch.randn(size=size)
        if device is not None:
            ret = ret.to(device)
        return torch.sqrt(a)* ret'''


def _between_minus_1_1_with_quantile(x, quantile, scale_to_minus_1_1 = True):
    # assume x is centred
    high_quantile = torch.quantile(x, quantile, dim = 0, interpolation='nearest')
    low_quantile = torch.quantile(x, 1 - quantile, dim = 0, interpolation='nearest')
    assert not (high_quantile < 0).any()
    assert not (low_quantile > 0).any()
    clamp_value = torch.max(torch.abs(high_quantile), torch.abs(low_quantile))
    clamp_value = clamp_value.unsqueeze(0).repeat(x.shape[0], *tuple([1]* len(x.shape[1:])))
    tmp = torch.clamp(x, min= - clamp_value, max=clamp_value)
    idx_high = (tmp >= clamp_value)
    idx_low = (tmp <= - clamp_value)
    tmp[idx_high] = clamp_value[idx_high]
    tmp[idx_low] = -clamp_value[idx_low]
    if scale_to_minus_1_1:
        tmp /= clamp_value
    else:
        # just clamp
        tmp = tmp.clamp(-1, 1)
        #tmp = tmp
    return tmp




''' They must have the same signature'''

def sample_2_gmm(n_samples, 
                 alpha = None, 
                 n = None, 
                 std = None, 
                 theta = 1.0, 
                 weights = None, 
                 device = None, 
                 normalize=False, 
                 isotropic = False,
                 between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    if weights is None:
        weights = np.array([0.5, 0.5])
    means = np.array([ [theta, 0], [-theta, 0] ])
    gmm = GaussianMixture(n_components=2)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(2)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff, scale_to_minus_1_1=True) # should do something with 1 / sqrt(n)
    return x[torch.randperm(x.size()[0])]

def sample_grid_gmm(n_samples, 
                    alpha = None, 
                    n = None, 
                    std = None, 
                    theta = None, 
                    weights = None, 
                    device = None, 
                    normalize=False, 
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    means = []
    for i in range(n):
        for j in range(n):
            means.append([i, j])
    means = np.array(means)
    gmm = GaussianMixture(n_components=n*n)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(n*n)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    return x[torch.randperm(x.size()[0])]


def gen_swiss_roll(n_samples, 
                   alpha = None, 
                   n = None, 
                   std = None, 
                   theta = None, 
                   weights = None, 
                   device = None, 
                   normalize=False, 
                   isotropic = False,
                   between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=std)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    return torch.tensor(x, dtype = torch.float32)


def sample_grid_sas(n_samples, 
                    alpha = 1.8, 
                    n = None, 
                    std = None, 
                    theta = 1.0, 
                    weights = None, 
                    device = None, 
                    normalize=False, 
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    data = std * gen_sas(alpha, size = (n_samples, 2), isotropic = isotropic)
    weights = np.concatenate((np.array([0.0]), weights))
    idx = np.cumsum(weights)*n_samples
    for i in range(n):
        for j in range(n):
            # for the moment just selecting exact proportions
            s = int(idx[i*n + j])
            e = int(idx[i*n + j + 1])
            data[s:e] = data[s:e] + torch.tensor([i, j])

    if normalize:
        data = (data - data.mean()) / data.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    #data = torch.tensor(data, dtype = torch.float32)
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff) # should do something with 1 / sqrt(n)
    return data[torch.randperm(data.size()[0])]

