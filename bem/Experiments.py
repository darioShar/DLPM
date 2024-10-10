import numpy as np
import torch
from .utils_exp import ExpUtils


'''
    Experiment class to manage experiments. Precise details and behaviour summarized in 
    corresponding github repo.
'''
class Experiment:
    ''' 
        Check if dic2 contains dic1
    '''
    @staticmethod
    def check_dict_eq(dic1, dic2):
        for k, v in dic1.items():
            if isinstance(v, dict):
                Experiment.check_dict_eq(v, dic2[k])
            elif isinstance(v, torch.Tensor):
                if (v != dic2[k]).any():
                    return False
            else:
                if v != dic2[k]:
                    return False
        return True

    '''
        check for device on which to run Pytorch
    '''
    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            device = "mps"
            mps_device = torch.device(device)
        elif torch.cuda.is_available():
            device = "cuda"
            cuda_device = torch.device(device)
        else:
            device = 'cpu'
            print ("GPU device not found.")
        print ('using device {}'.format(device))
        return device


    '''
        If running with cuda, perform some optimizations
    '''
    @staticmethod
    def _optimize_gpu(device):
        if device == 'cuda':
            #torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.backends.cudnn.benchmark = True

    '''
        Set random seeds for torch, numpy, and cuda torch
    '''
    @staticmethod
    def _set_seed(seed, device):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    '''
        Initialize the experiment. 
        checkpoint_dir is a directory path from where to load/save checkpoints
        p can be a directory or a path to a config file
        looger is a potential logger, e.g., NeptuneLogger()
    '''
    def __init__(self,
                 checkpoint_dir,
                 p,
                 logger = None,
                 config_path = None,
                 exp_hash = None,
                 eval_hash = None,
                 init_method_by_parameter = None,
                 init_models_by_parameter = None,
                 reset_models = None,
                 ):
        self.p = p
        self.logger = None
        self.config_path = config_path
        self.utils = ExpUtils(p, self.config_path, exp_hash, eval_hash, init_method_by_parameter, init_models_by_parameter, reset_models)
        self._reset_attributes(p, checkpoint_dir, logger)
        self._initialize()


    '''
        Reset the attributes of the experiment. Used to change parameters, checkpoint dir, or logger
    '''
    def _reset_attributes(self,
                         p,
                         checkpoint_dir,
                         logger):
        if (self.logger is not None) and (self.logger != logger) and (logger is not None):
            self.logger.stop()
            self.logger = logger
        elif (self.logger is None) and (logger is not None):
            self.logger = logger
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        if p is not None:
            self.utils.set_parameter(p, config_path=self.config_path)
            self.p = p
    

    '''
        Initialize the experiment. Set the device, seed, and perform optimizations
    '''
    def _initialize(self):
        device = self._get_device()
        self.utils.p['device'] = device
        self._optimize_gpu(device)
        if self.utils.p['seed'] is not None:
            self._set_seed(self.utils.p['seed'], device)

    def __del__(self):
        self.terminate()

    '''
        Terminate the experiment. Stop the logger
    '''
    def terminate(self):
        if self.logger is not None:
            self.logger.stop()
            self.logger = None

    '''
        Print the parameters of the running experiment
    '''
    def print_parameters(self):
        print('Training with the following parameters:')
        def print_dict(d, indent = 0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print('\t'*indent, k, '\t:')
                    print_dict(v, indent + 1)
                else:
                    print('\t'*indent, k, ':', v)
        print_dict(self.utils.p)

    '''
        Prepare the experiment. Load the model, data, test data, and manager
    '''
    def prepare(self, 
                p = None, 
                checkpoint_dir = None, 
                logger = None):
        self._reset_attributes(p, checkpoint_dir, logger)
        self.model, self.data, self.test_data, self.manager = self.utils.prepare_experiment(self.logger)

    '''
        Prepare and load an experiment from a checkpoint directory
        Prepare data, test data, and manager
        Retrieve metrics and model
    '''
    def load(self, 
             p = None,
             checkpoint_dir = None,
             logger = None,
             epoch = None,
             do_not_load_model = False,
             do_not_load_data = False,
             load_eval_subdir = None): # to load from subdir of new evaluations
        self._reset_attributes(p, checkpoint_dir, logger)
        self.model, self.data, self.test_data, self.manager = \
            self.utils.load_experiment(
                                       self.checkpoint_dir, 
                                       self.logger,
                                       curr_epoch=epoch,
                                       do_not_load_model = do_not_load_model,
                                       do_not_load_data=do_not_load_data,
                                       load_eval_subdir=load_eval_subdir)

    '''
        Save model, metrics to checkpoint directory
    '''
    def save(self, 
             curr_epoch = None,
             files='all',
             save_new_eval=False):
        return self.utils.save_experiment(
                            self.checkpoint_dir, 
                            self.manager, 
                            curr_epoch,
                            files = files,
                            save_new_eval=save_new_eval)

    '''
        Run the experiment. 
        Train the model for the specified number of epochs
        Evaluate the model and EMAs at specified intervals
        Save the model and metrics at specified intervals
        Then save the final model and metrics
        no_ema_eval: if True, do not evaluate the EMAs
        kwargs: parameters for the manager.train function
    '''
    def run(self, 
            no_ema_eval = False,
            **kwargs): # bs, lr, eval_freq, Lploss...
        
        epochs = self.utils.p['run']['epochs']
        def checkpoint_callback(curr_epoch):
            print('saved', self.save(curr_epoch=curr_epoch))

        self.manager.train(total_epoch=epochs, 
                           checkpoint_callback=checkpoint_callback,
                           no_ema_eval=no_ema_eval,
                           **kwargs)
        





