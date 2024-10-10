import os
import bem.Experiments as Exp
import bem.Logger as Logger
from bem.utils_exp import *
import dlpm.dlpm_experiment as dlpm_exp
from dlpm.NeptuneLogger import NeptuneLogger
from script_utils import *

def run_exp(config_path):
    args = parse_args()
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')

    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    checkpoint_dir = os.path.join('models', args.name)
    # the ExpUtils class specifies how to hash the parameter dict, and what and how to initiliaze methods and models
    exp = Exp.Experiment(checkpoint_dir=checkpoint_dir, 
                        p=p,
                        logger = NeptuneLogger() if args.log else None,
                        exp_hash= dlpm_exp.exp_hash, 
                        eval_hash=None, # will use default function
                        init_method_by_parameter= dlpm_exp.init_method_by_parameter,
                        init_models_by_parameter= dlpm_exp.init_models_by_parameter,
                        reset_models= dlpm_exp.reset_models)

    # load if necessary. Must be done here in case we have different hashes afterward
    if args.resume:
        if args.resume_epoch is not None:
            exp.load(epoch=args.resume_epoch)
        else:
            exp.load()
    else:
        exp.prepare()
    
    update_experiment_after_loading(exp, args) # update parameters after loading, like new optim learning rate...
    additional_logging(exp, args) # log some additional information
    exp.print_parameters() # print parameters to stdout
    
    # run the experiment
    exp.run(progress= p['run']['progress'],
            max_batch_per_epoch= args.n_max_batch, # to speed up testing
            no_ema_eval=args.no_ema_eval, # to speed up testing
        )
    
    # in any case, save last models.
    print(exp.save(curr_epoch=p['run']['epochs']))
    
    # terminates everything (e.g., logger etc.)
    exp.terminate()


if __name__ == '__main__':
    config_path = 'dlpm/configs/'
    run_exp(config_path)