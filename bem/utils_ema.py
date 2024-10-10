import torch.nn as nn
import copy

class EMAHelper(object):
    def __init__(self, 
                 model,
                 mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.register(model)
        self.model = copy.deepcopy(model) # keep a copy of the model

    # create copy of trainable parameters
    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    # updates trainable parameters
    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data.detach() + self.mu * self.shadow[name].data

    # copy ema parameters to model
    def _ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    # create a copy of the model with ema parameters
    # NYI: not our constructor
    def _ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy
    
    # just return a usable ema model
    def get_ema_model(self):
        self._ema(self.model)
        return self.model

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
