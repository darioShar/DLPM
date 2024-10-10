import os
import bem.Experiments as Exp
from bem.utils_exp import *
import dlpm.dlpm_experiment as dlpm_exp
from script_utils import *
from bem.utils_exp import FileHandler

def edit_exp(config_folder):
    dataset = 'mnist'
    exp_name = 'mnist_new_exp'

    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_folder, dataset + '.yml')
    
    to_replace = {
        'nblocks': 8, 
        'time_emb_size': 32,
        'time_emb_type': 'learnable',
    }

    for k, v in to_replace.items():
        p['model'][k] = v

        # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    checkpoint_dir = os.path.join('models', exp_name)
    # the ExpUtils class specifies how to hash the parameter dict, and what and how to initiliaze methods and models
    exp = Exp.Experiment(checkpoint_dir=checkpoint_dir, 
                        p=p,
                        logger = None,
                        exp_hash= dlpm_exp.exp_hash, 
                        eval_hash= None, # will use default function
                        init_method_by_parameter= dlpm_exp.init_method_by_parameter,
                        init_models_by_parameter= dlpm_exp.init_models_by_parameter,
                        reset_models= dlpm_exp.reset_models)
    


    # load if necessary. Must be done here in case we have different hashes afterward
    load_epoch = 100
    exp.load(epoch=load_epoch)
    
    to_replace = {
        'nblocks': 4, 
        'time_emb_size': 64,
        'time_emb_type': 'learnable',
    }

    for k, v in to_replace.items():
        exp.p['model'][k] = v
    
    exp.print_parameters()

    new_models, _, _, new_manager = exp.utils.prepare_experiment(exp.p)
    #exp.manager.model, exp.manager.optimizer, exp.manager.learning_schedule = reset_model(exp.p)
    exp.manager = new_manager
    exp.manager.models = new_models

    print(exp.save())
    print(exp.save(curr_epoch = load_epoch))


if __name__ == '__main__':
    config_folder = 'dlpm/configs/'
    edit_exp(config_folder)

