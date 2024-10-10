import numpy as np
import torch
import pyemd

# to compute W_p loss
def compute_mean_lploss(tens, lploss = 2.):
            # simply flatten the tensor
            return torch.pow(torch.linalg.norm(tens.flatten(), 
                                               ord = lploss,), 
                                               #dim = list(range(0, len(tens.shape)))), 
                                                lploss) \
                / torch.prod(torch.tensor(tens.shape), dim = 0)

# updated to manually compute bins if requested
def compute_wasserstein_distance(data, 
                                 gen_samples,
                                 manual_compute = False,
                                 num_samples = -1, 
                                 distance='euclidean',
                                 normalized=True,
                                 bins='auto',
                                 _range=None):
    if manual_compute:
        # each vector is a histogram fof bins
        # that is the density of the corresponding data in each bin.
        # pairwise distance is computed between each bin
        # thus if each data point has its own bin, we need 2*N bins
        # and a 4*N*N matrix.
        # use L1 loss
        lploss = 1.
        N = np.min((data.shape[0], gen_samples.shape[0])) if num_samples == -1 else num_samples
        # equal histograms, first bins for data, second one for gen_samples
        data_1_array = np.concatenate((np.ones(N) / N, np.zeros(N)),dtype = np.float64)
        data_2_array = np.concatenate((np.zeros(N), np.ones(N) / N),dtype = np.float64)
        distance_data = np.zeros((2*N, 2*N), dtype = np.float64)
        for i in range(2*N):
            for j in range(2*N):
                #distance_data = np.array([[compute_loss_all(data1[i] - data2[j]) for j in range(data2.shape[0])] for i in range(data1.shape[0])])
                data_i = data[i] if i < N else gen_samples[i % N]
                data_j = data[j] if j < N else gen_samples[j % N]
                distance_data[i, j] = compute_mean_lploss(data_i - data_j, lploss = lploss)
        res = pyemd.emd(data_1_array,
                        data_2_array,
                        np.float64(distance_data))
        res = res**(lploss)
        return res
    return pyemd.emd_samples(gen_samples[:num_samples], 
                             data[:num_samples],
                            distance = distance, 
                            normalized=normalized, 
                            bins=bins, 
                            range=_range)