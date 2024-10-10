import numpy as np
from .prd_score import compute_prd_from_embedding, prd_to_max_f_beta_pair


#num_angles = 501
def compute_precision_recall_curve(data, gen_samples, num_angles = 201, num_clusters = 20):
    # flatten last dimensions
    nsamples = data.shape[0]
    data = data.view(nsamples, -1)
    data_gen = gen_samples.view(nsamples, -1)
    return compute_prd_from_embedding(data, data_gen, num_angles=num_angles, num_clusters = num_clusters)

def compute_f_beta(prec, rec):
    a, b = prd_to_max_f_beta_pair(prec, rec)
    return np.array([a, b])