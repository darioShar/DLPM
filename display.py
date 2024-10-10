import os
import bem.Experiments as Exp
import bem.Logger as Logger
from bem.utils_exp import *
import dlpm.dlpm_experiment as dlpm_exp
from dlpm.NeptuneLogger import NeptuneLogger
from script_utils import *
import matplotlib.pyplot as plt


SAVE_ANIMATION_PATH = './animation'


def display_exp(config_path):
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

    exp.prepare()
        
    additional_logging(exp, args)

    # print parameters
    exp.print_parameters()

    print('Loading latest model')
    exp.load()
    
    update_experiment_after_loading(exp, args)

    # some information
    run_info = [exp.p['data']['dataset'], exp.manager.method.reverse_steps, exp.manager.total_steps]
    title = '{}, reverse_steps={}, training_steps={}'.format(*run_info[:3])
    
    # display plot and animation, for a specific model
    anim = exp.manager.display_plots(ema_mu=None, # can specify ema rate, if such a model has been trained
                                plot_original_data=False, 
                                title=title,
                                nb_datapoints=20000 if args.generate is None else args.generate, # number of points to display.
                                marker='.', # '.' marker displays pixel-wide points.
                                color='blue', # color of the points
                                xlim = (-1, 2.5), # x-axis limits
                                ylim = (-1, 2.5), # y-axis limits
                                alpha = 1.0,
                                )
    
    # save animation
    path = os.path.join(SAVE_ANIMATION_PATH, '_'.join([str(x) for x in run_info]))
    anim.save(path + '.mp4')
    print('Animation saved in {}'.format(path))

    # stops the thread from continuing
    plt.show()

    # close everything
    exp.terminate()

if __name__ == '__main__':
    config_path = 'dlpm/configs/'
    display_exp(config_path)