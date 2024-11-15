import neptune
from neptune.utils import stringify_unsupported
import torch
import numpy as np
from bem.Logger import Logger
from .private_code import Neptune_Project, Neptune_API_key



''' this will manage all the information that could be contained in dictionnary like this:
p = {
    'settings' : {'lr' : ..., 'optimizer': ...},
    'model' : {},
    'data' : {},
    'eval' : {}
}
'''

class NeptuneLogger(Logger): 
    def __init__(self, p = None):
        super().__init__()
        # connect
        self.run = neptune.init_run(project=Neptune_Project, api_token=Neptune_API_key ,capture_hardware_metrics = True)
        if p is not None:
            self.initialize(p)
    
    def initialize(self, p):
        for key in p.keys():
            self.run[key] = stringify_unsupported(p[key])

    def set_values(self, value_dict):
        print('setting values in neptune logger')
        def aux(current_str, dic):
            for k, v in dic.items():
                new_str = '/'.join([current_str, k]) if current_str != '' else k
                if isinstance(v, dict):
                    aux(new_str, v)
                else:
                    if self.run.exists(new_str):
                        del self.run[new_str]
                    
                    if not isinstance(v, (list, np.ndarray, torch.Tensor)):
                        print('setting {} = {}'.format(new_str, v))
                        self.run[new_str].append(v)
                    else:
                        #self.run[new_str].extend(v)
                        print('setting {} of size {}'.format(new_str, len(v)))
                        if len(v) >= 1000: # surely must be array of float
                            self.run[new_str].extend([x for x in v])
                        else:
                            for v_data in v:
                                self.run[new_str].append(v_data)
                    #self.run[current_str] = stringify_unsupported(dic[k])
        aux('', value_dict)

    def log(self, data_type, data):
        self.run['/'.join(['eval', data_type])].append(data)

    def stop(self):
        self.run.stop()

