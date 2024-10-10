import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from transformers import get_scheduler
import torchvision.utils as tvu

from pathlib import Path
import yaml
import os
import hashlib

from .datasets import get_dataset, is_image_dataset
from .TrainingManager import TrainingManager
from .evaluate.EvaluationManager import EvaluationManager
from .GenerationManager import GenerationManager
from .datasets import inverse_affine_transform

''''''''''' FILE MANIPULATION '''''''''''

class FileHandler:
    '''
    p: parameters dictionnary
    '''
    def __init__(self, 
                 exp_hash = None, 
                 eval_hash = None):
        self.exp_hash = exp_hash if exp_hash is not None else FileHandler.default_exp_hash
        self.eval_hash = eval_hash if eval_hash is not None else FileHandler.default_eval_hash
    
    # by default returns the parameter dictionnary only relative to the selected method.
    # should be overriden for specific needs
    @staticmethod
    def default_exp_hash(p):
        return {'data': p['data'],
                p['method']: p[p['method']],
                'model': p['model'][p['method']],
            }
    @staticmethod
    def default_eval_hash(p):
        return  {'eval': p['eval'][p['method']]}


    def get_exp_hash(self, p, verbose=False):
        # tmp = 'exp_{}_{}_{}'.format(p['method'], p['data']['dataset'], datetime.now().strftime('%d_%m_%y_%H_%M_%S'))
        to_hash = self.exp_hash(p) # model_param_to_use(p)
        res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        res = str(res)[:16]
        #res = str(hex(abs(hash(tuple(p)))))[2:]
        if verbose:
            print('experiment hash: {} \n\tParameters: {}'.format(res, to_hash))
        return res

    # this is an evaluation only hash
    def get_eval_hash(self, p, verbose = False):
        to_hash = self.eval_hash(p)
        res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        res = str(res)[:8]
        if verbose:
            print('eval hash: {} \n\tParameters: {}'.format(res, to_hash))
        return res

    # returns new save folder and parameters hash
    def get_exp_path_from_param(self,
                                 p,
                                folder_path, 
                                make_new_dir = False):
        
        h = self.get_exp_hash(p)
        save_folder_path = os.path.join(folder_path, p['data']['dataset'])
        if make_new_dir:
            Path(save_folder_path).mkdir(parents=True, exist_ok=True)
        return save_folder_path, h

    # returns eval folder given save folder, and eval hash
    def get_eval_path_from_param(self, 
                                      p,
                                save_folder_path, 
                                make_new_dir = False):
        h = self.get_exp_hash(p)
        h_eval = self.get_eval_hash(p)
        eval_folder_path = os.path.join(save_folder_path, '_'.join(('new_eval', h, h_eval)))
        if make_new_dir:
            Path(eval_folder_path).mkdir(parents=True, exist_ok=True)
        return eval_folder_path, h, h_eval

    # returns paths for model and param
    # from a base folder. base/data_distribution/
    def get_paths_from_param(self, 
                             p,
                            folder_path, 
                            make_new_dir = False, 
                            curr_epoch = None, 
                            new_eval_subdir=False,
                            do_not_load_model=False): # saves eval and param in a new subfolder
        save_folder_path, h = self.get_exp_path_from_param(p, folder_path, make_new_dir)
        if new_eval_subdir:
            eval_folder_path, h, h_eval = self.get_eval_path_from_param(p, save_folder_path, make_new_dir)

        names = ['model', 'parameters', 'eval']
        # create path for each name
        # in any case, model get saved in save_folder_path
        if curr_epoch is not None:
            L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
        else:
            L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
            if not do_not_load_model:
                # checks if model is there. otherwise, loads latest model. also checks equality of no_iteration model and latest iteration one
                # list all model iterations
                model_paths = list(Path(save_folder_path).glob('_'.join(('model', h)) + '*'))
                assert len(model_paths) > 0, 'no models to load in {}, with hash {}'.format(save_folder_path, h)
                max_model_iteration = 0
                max_model_iteration_path = None
                for i, x in enumerate(model_paths):
                    if str(x)[:-3].split('_')[-1].isdigit() and (len(str(x)[:-3].split('_')[-1]) < 8): # if it is digit, and not hash
                        model_iter = int(str(x)[:-3].split('_')[-1])
                        if max_model_iteration< model_iter:
                            max_model_iteration = model_iter
                            max_model_iteration_path = str(x)
                if max_model_iteration_path is not None:
                    if Path(L['model'] + '.pt').exists():
                        print('Found another save with no specified iteration alonside others with specified iterations. Will not load it')
                    print('Loading trained model at iteration {}'.format(max_model_iteration))
                    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(max_model_iteration)])}
                elif Path(L['model']+ '.pt').exists():
                    print('Found model with no specified iteration. Loading it')
                    # L already holds the right name
                    #L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
                else:
                    raise Exception('Did not find a model to load at location {} with hash {}'.format(save_folder_path, h))
                
        # then depending on save_new_eval, save either in save_folder or eval_folder
        if new_eval_subdir:
            if curr_epoch is not None:
                L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval, str(curr_epoch)]) for name in names[1:]})
            else:
                L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval]) for name in names[1:]})
        else:
            # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
            # so we do not append curr_epoch here. 
            L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
        
        return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval


        #save_folder_path, h = get_hash_path_from_param(p, folder_path, make_new_dir)
        #if new_eval_subdir:
        #    eval_folder_path, h, h_eval = get_eval_path_from_param(p, save_folder_path, make_new_dir)
    #
        #names = ['model', 'parameters', 'eval']
        ## create path for each name
        ## in any case, model get saved in save_folder_path
        #if curr_epoch is not None:
        #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
        #else:
        #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
        ## then depending on save_new_eval, save either in save_folder or eval_folder
        #if new_eval_subdir:
        #    if curr_epoch is not None:
        #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval, str(curr_epoch)]) for name in names[1:]})
        #    else:
        #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval]) for name in names[1:]})
        #else:
        #    # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
        #    # so we do not append curr_epoch here. 
        #    L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
        #
        #return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval


    def _prepare_data_directories(self, dataset_name, dataset_files, remove_existing_eval_files, num_real_data, hash_params):

        if dataset_files is None:
            # do nothing, assume no data will be generated
            print('(prepare data directories) assuming no data will be generated.')
            return None, None

        # create directory for saving images
        folder_path = os.path.join('eval_files', dataset_name)
        generated_data_path = os.path.join(folder_path, 'generated_data', hash_params)
        if not is_image_dataset(dataset_name):
            # then we have various versions of the same dataset
            real_data_path = os.path.join(folder_path, 'original_data', hash_params)
        else:
            real_data_path = os.path.join(folder_path, 'original_data')
        
        #Path(generated_data_path).mkdir(parents=True, exist_ok=True)
        #Path(real_data_path).mkdir(parents=True, exist_ok=True)

        def remove_file_from_directory(dir):
            # remove the directory
            if not dir.is_dir():
                raise ValueError(f'{dir} is not a directory')
            # print('removing files in directory', dir)
            for file in dir.iterdir():
                file.unlink()

        def save_images(path):
                print('storing dataset in', path)
                # now saving the original data
                assert dataset_name.lower() in ['mnist', 'cifar10', 'celeba'], 'only mnist, cifar10, celeba datasets are supported for the moment. \
                    For the moment we are loading {} data points. We may need more for the other datasets, \
                        and anyway we should implement somehting more systematic'.format(num_real_data)
                #data = gen_model.load_original_data(evaluation_files) # load all the data. Number of datapoints specific to mnist and cifar10
                data_to_store = num_real_data
                print('saving {} original images from pool of {} datapoints'.format(data_to_store, len(dataset_files)))
                for i in range(data_to_store):
                    if (i%500) == 0:
                        print(i, end=' ')
                    tvu.save_image(inverse_affine_transform(dataset_files[i][0]), os.path.join(path, f"{i}.png"))
        
        path = Path(generated_data_path)
        if path.exists():
            if remove_existing_eval_files:
                remove_file_from_directory(path)
        else:
            path.mkdir(parents=True, exist_ok=True)

        path = Path(real_data_path)
        if is_image_dataset(dataset_name):
            if path.exists():
                print('found', path)
                assert path.is_dir(), (f'{path} is not a directory')
                # check that there are the right number of image files, else remove and regenerate
                if len(list(path.iterdir())) != num_real_data:
                    remove_file_from_directory(path)
                    save_images(path)
            else:
                path.mkdir(parents=True, exist_ok=True)
                save_images(path)
        else:
            if path.exists():
                remove_file_from_directory(path)
            else:
                path.mkdir(parents=True, exist_ok=True)

        return generated_data_path, real_data_path

    def prepare_data_directories(self, p, dataset_files):
        # prepare the evaluation directories
        return self._prepare_data_directories(dataset_name=p['data']['dataset'],
                                dataset_files = dataset_files, 
                                remove_existing_eval_files = False if p['eval']['data_to_generate'] == 0 else True,
                                num_real_data = p['eval']['real_data'],
                                hash_params = '_'.join([self.get_exp_hash(p), self.get_eval_hash(p)]), # for saving images. We want a hash specific to the training, and to the sampling
                                )
    @staticmethod
    def get_param_from_config(config_path, config_file):
        with open(os.path.join(config_path, config_file), "r") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    # loads all params from a specific folder
    def get_params_from_folder(folder_path):
        return [torch.load(path) for path in Path(folder_path).glob("parameters*")]



class InitUtils:

    def __init__(self, 
                 init_method_by_parameter,
                 init_models_by_parameter,
                 reset_models
                 ):
        assert init_method_by_parameter is not None, 'init_method_by_parameter should be defined'
        assert init_models_by_parameter is not None, 'init_models_by_parameter should be defined'
        assert reset_models is not None, 'reset_models should be defined'

        self.init_method_by_parameter = init_method_by_parameter
        self.init_models_by_parameter = init_models_by_parameter
        self.reset_models = reset_models

    # @abstractmethod
    # def init_method_by_parameter(self):
    #     pass

    # @abstractmethod
    # def init_models_by_parameter(self):
    #     pass

    # @abstractmethod
    # def reset_models(self):
    #     pass
    
    # we propose a default initialization for the optimizer
    @staticmethod
    def init_default_optimizer(p, model):
        # training manager
        optimizer = optim.AdamW(model.parameters(), 
                                lr=p['optim']['lr'], 
                                betas=(0.9, 0.999)) # beta_2 0.95 instead of 0.999
        return optimizer
    
    # we propose a default initialization for the learning schedule
    @staticmethod
    def init_default_ls(p, optim):
        if p['optim']['schedule'] == None:
            return None
        
        if p['optim']['schedule'] == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size= p['optim']['lr_step_size'], 
                                                        gamma= p['optim']['lr_gamma'], 
                                                        last_epoch=-1)
        else: 
            lr_scheduler = get_scheduler(
                p['optim']['schedule'],
                # "cosine",
                # "cosine_with_restarts",
                optimizer=optim,
                num_warmup_steps=p['optim']['warmup'],
                num_training_steps=p['optim']['lr_steps'],
            )
        return lr_scheduler

    def init_data_by_parameter(self, p):
        # get the dataset
        dataset_files, test_dataset_files = get_dataset(p)
        
        # implement DDP later on
        data = DataLoader(dataset_files, 
                        batch_size=p['training']['batch_size'], 
                        shuffle=True, 
                        num_workers=p['training']['num_workers'])
        test_data = DataLoader(test_dataset_files,
                                batch_size=p['training']['batch_size'],
                                shuffle=True,
                                num_workers=p['training']['num_workers'])
        return data, test_data, dataset_files, test_dataset_files

    def init_experiment(self,
                        p, 
                        data,
                        test_data,
                        gen_data_path, 
                        real_data_path, 
                        logger = None):

        # initialize models
        models, optimizers, learning_schedules = self.init_models_by_parameter(p)

        # intialize logger
        if logger is not None:
            logger.initialize(p)

        # get noising process
        method = self.init_method_by_parameter(p)
        
        # get generation manager
        kwargs = p['eval'][p['method']]# here kwargs is passed to the underlying Generation Manager.
        gen_manager = GenerationManager(method, 
                                    dataloader=data, 
                                    is_image = is_image_dataset(p['data']['dataset']),
                                    **kwargs)

        # run evaluation on train or test data
        eval = EvaluationManager( 
                method=method,
                gen_manager=gen_manager,
                dataloader=data,
                verbose=True, 
                logger = logger,
                data_to_generate = p['eval']['data_to_generate'],
                batch_size = p['eval']['batch_size'],
                is_image = is_image_dataset(p['data']['dataset']),
                gen_data_path=gen_data_path,
                real_data_path=real_data_path
        )
        
        # run training
        kwargs = p['training'][p['method']] # here kwargs goes to manager (ema_rates), train_loop (grad_clip), and eventually to training_losses (monte_carlo...)
        manager = TrainingManager(models,
                    data,
                    method,
                    optimizers,
                    learning_schedules,
                    eval,
                    logger,
                    reset_models=self.reset_models,
                    eval_freq = p['run']['eval_freq'],
                    checkpoint_freq = p['run']['checkpoint_freq'],
                    # ema_rate, grad_clip
                    **kwargs
                    )
        return models, manager



''''''''''' FULL CLASS + LOADING/SAVING '''''''''''

class ExpUtils(FileHandler, InitUtils):

    def __init__(self, 
                 p,
                 config_path = None,
                 exp_hash = None,
                 eval_hash = None,
                 init_method_by_parameter = None,
                 init_models_by_parameter = None,
                 reset_models = None,
                 ):
        
        FileHandler.__init__(self, exp_hash, eval_hash)
        InitUtils.__init__(self, init_method_by_parameter, init_models_by_parameter, reset_models)
        # self.file_handler_class = file_handler_class
        # self.init_utils_class = init_utils_class
        # self.file_handler = None
        # self.init_utils = None
        self.set_parameter(p, config_path = config_path)
    
    '''
        Set the parameters of the experiment. p can be a path to a config file or a dictionary of parameters
    '''
    def set_parameter(self, p, config_path = None):
        if isinstance(p, str): # config file
            self.p = FileHandler.get_param_from_config(config_path, p)
        elif isinstance(p, dict): # dictionnary
            self.p = p
        else:
            raise Exception('p should be a path to a config file or a dictionary of parameters. Got {}'.format(p))
        
        # if self.file_handler is None:
        #     self.file_handler = self.file_handler_class(self.p)
        # else: 
        #     self.file_handler.p = self.p
        # if self.init_utils is None:
        #     self.init_utils = self.init_utils_class(self.p)
        # else:
        #     self.init_utils.p = self.p


    def prepare_experiment(self, logger = None, do_not_load_data=False):
        # prepare the evaluation directories
        if do_not_load_data:
            data, test_data, dataset_files, test_dataset_files = None, None, None, None
        else:
            data, test_data, dataset_files, test_dataset_files = self.init_data_by_parameter(self.p)
        gen_data_path, real_data_path = self.prepare_data_directories(self.p, dataset_files)
        models, manager =  self.init_experiment(self.p, data, test_data, gen_data_path, real_data_path, logger)
        return models, data, test_data, manager

    def _load_experiment(self, 
                        model_path, 
                        eval_path, 
                        logger,
                        do_not_load_model = False,
                        do_not_load_data = False):
        models, data, test_data, manager = self.prepare_experiment(logger, do_not_load_data)
        if not do_not_load_model:
            print('loading from model file {}'.format(model_path))
            manager.load(model_path)
        print('loading from eval file {}'.format(eval_path))
        manager.load_eval_metrics(eval_path)
        #manager.losses = torch.load(eval_path)
        return models, data, test_data, manager

    # loads a model from some param as should be contained in folder_path.
    # Specify the training epoch at which to load; defaults to latest
    def load_experiment(self, 
                        folder_path, 
                        logger=None,
                        curr_epoch = None,
                        do_not_load_model = False,
                        do_not_load_data=False,
                        load_eval_subdir=False):
        model_path, _, eval_path = self.get_paths_from_param(self.p, 
                                                    folder_path, 
                                                    curr_epoch=curr_epoch,
                                                    new_eval_subdir = load_eval_subdir,
                                                    do_not_load_model=do_not_load_model)
        models, data, test_data, manager = self._load_experiment( 
                                                model_path, 
                                                eval_path, 
                                                logger,
                                                do_not_load_model=do_not_load_model,
                                                do_not_load_data=do_not_load_data)
        return models, data, test_data, manager


    # unique hash of parameters, append training epochs
    # simply separate folder by data distribution and alpha value
    def save_experiment(self, 
                        base_path, 
                        manager,
                        curr_epoch = None,
                        files = 'all',
                        save_new_eval=False): # will save eval and param in a subfolder.
        if isinstance(files, str):
            files = [files]
        for f in files:
            assert f in ['all', 'model', 'eval', 'param'], 'files must be one of all, model, eval, param'
        model_path, param_path, eval_path = self.get_paths_from_param(self.p, 
                                                                base_path, 
                                                                make_new_dir = True, 
                                                                curr_epoch=curr_epoch,
                                                                new_eval_subdir=save_new_eval)
        #model_path = '_'.join([model_path, str(manager.training_epochs())]) 
        #losses_path = '_'.join([model_path, 'losses']) + '.pt'
        if 'all' in files:
            manager.save(model_path)
            manager.save_eval_metrics(eval_path)
            torch.save(self.p, param_path)
            return model_path, param_path, eval_path
        
        # else, slightly more complicated logic
        objects_to_save = {name: {'path': path, 'saved':False} for name, path in zip(['model', 'eval', 'param'],
                                                                        [model_path, eval_path, param_path])}
        for name, obj in objects_to_save.items():
            if name in files:
                obj['saved'] = True
                if name == 'model':
                    manager.save(obj['path'])
                elif name == 'eval':
                    manager.save_eval_metrics(obj['path'])
                elif name == 'param':
                    torch.save(self.p, obj['path'])
        
        # return values in the right order
        return tuple(objects_to_save[name]['path'] if objects_to_save[name]['saved'] else None for name in ['model', 'eval', 'param'])

        if 'model' in files:
            manager.save(model_path)
            eval_path = None
            param_path = None
        if 'eval' in files:
            manager.save_eval_metrics(eval_path)
            model_path = None
            param_path = None
        if 'param' in files:
            torch.save(p, param_path)
            model_path = None
            eval_path = None
        return model_path, param_path, eval_path



