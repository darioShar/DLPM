from bem.datasets import get_dataset, is_image_dataset
import dlpm.models.Model as Model
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess
import dlpm.models.unet as unet
from bem.utils_exp import InitUtils
from torch.utils.data import DataLoader



''''''''''' FILE MANIPULATION '''''''''''
def exp_hash(p):
    model_param = p['model']
    to_hash = {'data': {k:v for k, v in p['data'].items() if k in ['dataset', 'channels', 'image_size']},
            p['method']: {k:v for k, v in p[p['method']].items()},
            'model':  {k:v for k, v in model_param.items()}, 
            #'optim': p['optim'],
            #'training': p['training']
            }
    return to_hash
    

''' INITIALIZATION UTILS '''
# for the moment, only unconditional models
def _unet_model(p):
    # the classical channel multiplier. Can choose otherwise in config files.
    '''
    image_size = p['data']['image_size']
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    '''

    channels = p['data']['channels']
    out_channels = channels
    p_model_unet = p['model']
    
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=p_model_unet['model_channels'],
            out_channels= out_channels,
            num_res_blocks=p_model_unet['num_res_blocks'],
            attention_resolutions=p_model_unet['attn_resolutions'],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= p_model_unet['dropout'],
            channel_mult= p_model_unet['channel_mult'], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,
            num_heads=p_model_unet['num_heads'],
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
        )
    return model
    
# LIM models
def _lim_ddpm_unet(p):
    import dlpm.models.ddpm as LIM_DDPM_Unet
    return LIM_DDPM_Unet.Model(p)

def _lim_ncsnpp(p):
    import dlpm.models.ncsnpp as LIM_NCSNPP
    return LIM_NCSNPP.NCSNpp(p)

def init_model_by_parameter(p):
    # model
    if not is_image_dataset(p['data']['dataset']):
        model = Model.MLPModel(p)
    else:
        if p['model']['model_type'] == 'ddpm':
            print('Using improved DDPM Unet implementation')
            model = _unet_model(p)
        elif p['model']['model_type'] == 'lim_ddpm':
            print('Using LIM DDPM Unet implementation')
            model = _lim_ddpm_unet(p)
        elif p['model']['model_type'] == 'lim_ncsnpp':
            print('Using LIM NCSNPP implementation')
            model = _lim_ncsnpp(p)
        else:
            raise ValueError('model type {} not recognized'.format(p['model']['model_type']))
    return model.to(p['device'])


def init_data_by_parameter(p):
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


def init_method_by_parameter(p):
    chosen_gen_model = p['method']
    assert chosen_gen_model in ['dlpm', 'lim'], f"chosen_gen_model should be in ['dlpm', 'lim'], got {chosen_gen_model}"
    LIM = (chosen_gen_model == 'lim')
    if LIM:
        method = GenerativeLevyProcess(alpha = p[chosen_gen_model]['alpha'],
                                    device = p['device'],
                                    reverse_steps = p[chosen_gen_model]['reverse_steps'],
                                    rescale_timesteps = p[chosen_gen_model]['rescale_timesteps'],
                                    isotropic = p[chosen_gen_model]['isotropic'],
                                    # clamp_a=p['training'][chosen_gen_model]['clamp_a'],
                                    # clamp_eps=p['training'][chosen_gen_model]['clamp_eps'],
                                    LIM = LIM, # use continuous LIM
        )
    else:
        method = GenerativeLevyProcess(alpha = p[chosen_gen_model]['alpha'],
                                    device = p['device'],
                                    reverse_steps = p[chosen_gen_model]['reverse_steps'],
                                    model_mean_type = p[chosen_gen_model]['mean_predict'],
                                    model_var_type = p[chosen_gen_model]['var_predict'],
                                    rescale_timesteps = p[chosen_gen_model]['rescale_timesteps'],
                                    isotropic = p[chosen_gen_model]['isotropic'],
                                    # clamp_a=p['training'][chosen_gen_model]['clamp_a'],
                                    # clamp_eps=p['training'][chosen_gen_model]['clamp_eps'],
                                    LIM = LIM, # do not use continuous LIM
                                    scale = p[chosen_gen_model]['scale'],
                                    input_scaling=p[chosen_gen_model]['input_scaling'],
        )
    return method

def init_optimizer_by_parameter(p, model):
    return InitUtils.init_default_optimizer(p, model)

def init_ls_by_parameter(p, optim):
    return InitUtils.init_default_ls(p, optim)

def init_models_by_parameter(p):
    models = {
        'default': init_model_by_parameter(p)
        }
    optimizers = {
        'default': init_optimizer_by_parameter(p, models['default']),
        }
    learning_schedules = {
        'default': init_ls_by_parameter(p, optimizers['default'])
        }

    return models, optimizers, learning_schedules

def reset_models(p):
    return init_models_by_parameter(p)

