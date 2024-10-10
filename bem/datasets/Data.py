import numpy as np
from inspect import signature
from torch.utils.data import Dataset

from .Distributions import *
from .torchlevy.levy import LevyStable


# Sample from useful data distributions like swiss roll, GMM, levy variables

# Wrapper class to call all the generation functions
# We can specify arbitrary *args and **kwargs. They must match with the selected generation
# function of course.

# TODO: For the moment, better to just give kwargs, as one could not necesarrily remember 
# the *args order...
class Generator(Dataset):

    available_distributions = [
        'gmm_2',
        'gmm_grid',
        'swiss_roll',
        'skewed_levy',
        'sas',
        'sas_grid'
    ]
    
    def __init__(self, operation, transform = None, *args, **kwargs):
        self.transform = lambda x: x
        if transform is not None:
            self.transform = transform
        self.kwargs = kwargs
        self.args = args
        self.samples = None
        self.levy_stable = LevyStable()
        self.available_distributions_dict = \
        {'gmm_2': sample_2_gmm,
        'gmm_grid': sample_grid_gmm,
        'swiss_roll':gen_swiss_roll,
        'skewed_levy': gen_skewed_levy, # self.levy_stable.gen_skewed_levy,#gen_skewed_levy,
        'sas': gen_sas,#self.levy_stable.gen_sas,#gen_sas,
        #'gaussian_noising': gaussian_noising,
        #'stable_noising': stable_noising,
        'sas_grid': sample_grid_sas
        }


        try:
            self.generator = self.available_distributions_dict[operation]
        except:
            raise Exception('Unknown distribution to sample from. \
            Available distributions: {}'.format(list(self.available_distributions_dict.keys())))
    
    
    def setTransform(self, transform):
        self.transform = transform
    
    # replaces missing elements of args and kwargs by those of self.args and self.kwargs
    # replaces elements of kwargs totally if non void
    def setParams(self, *args, **kwargs):
        if args == () and kwargs == {}:
            raise Exception('Given void parameters')
        
        self.args = tuple(map(lambda x, y: y if y is not None else x, self.args, args))
        self.kwargs.update(kwargs)

    def getSignature(self):
        return signature(self.generator)
    
    def getName(self):
        return 
    
    # generate by replacing self.args by args if it is not ()
    # and replace potentially missing kwargs by thos provided.
    def generate(self, *args, **kwargs):
        tmp_kwargs = self.kwargs | kwargs        
        if args == () and kwargs == {} and self.kwargs == {}:
            raise Exception('No parameters for data generation')
        if args == ():
            self.samples = self.transform(self.generator(*self.args, **tmp_kwargs))
            return self.samples
        self.samples = self.transform(self.generator(*args, **tmp_kwargs))
        return self.samples

    def __len__(self):
        return self.samples.size()
    
    def __getitem__(self, idx):
        return self.samples[idx]
    