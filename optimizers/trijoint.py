import torch
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from torch.nn.utils.clip_grad import clip_grad_norm

class Trijoint(torch.optim.Optimizer):

    def __init__(self, opt, model, engine=None):
        self.model = model
        self.lr = opt['lr']
        self.switch_epoch = opt['switch_epoch']
        self.clip_grad = opt.get('clip_grad', False)
        self.optimizers = {}
        self.optimizers['recipe'] = torch.optim.Adam(self.model.network.get_parameters_recipe(), self.lr)
        self.optimizers['image'] = torch.optim.Adam(self.model.network.get_parameters_image(), self.lr)
        self.current_optimizer_name = 'recipe'
        self.epoch = 0
        self._activate_model()
        if engine:
            engine.register_hook('train_on_start_epoch', self._auto_fixed_fine_tune)
            if self.clip_grad:
                engine.register_hook('train_on_print', self.print_total_norm)

    def state_dict(self):
        state = {}
        state['optimizers'] = {}
        for key, value in self.optimizers.items():
            state['optimizers'][key] = value.state_dict()
        state['attributs'] = {
            'current_optimizer_name': self.current_optimizer_name,
            'epoch': self.epoch
        }
        return state

    def load_state_dict(self, state_dict):
        for key, _ in self.optimizers.items():
            value = state_dict['optimizers'][key]
            if len(value['state']) != 0: # bug pytorch??
                self.optimizers[key].load_state_dict(value)
        if 'attributs' in state_dict:
            for key, value in state_dict['attributs'].items():
                setattr(self, key, value)
        self._activate_model()

    def zero_grad(self):
        for name in self.current_optimizer_name.split('&'):
            self.optimizers[name].zero_grad()

    def step(self, closure=None):
        if self.clip_grad:
            self.clip_grad_norm()
        for name in self.current_optimizer_name.split('&'):
            self.optimizers[name].step(closure)

    def clip_grad_norm(self):
        params = []
        for k in self.optimizers:
            for group in self.optimizers[k].param_groups:
                for p in group['params']:
                    params.append(p)
        self.total_norm = clip_grad_norm_(params, self.clip_grad).item()
        Logger().log_value('optimizer.total_norm', self.total_norm, should_print=False)

    def print_total_norm(self):
        Logger()('{}  total_norm: {:.6f}'.format(' '*len('train'), self.total_norm))

    def _activate_model(self):
        optim_name = self.current_optimizer_name
        activate_recipe = (optim_name == 'recipe') or (optim_name == 'recipe&image')
        activate_image = (optim_name == 'image') or (optim_name == 'recipe&image')
        for p_dict in self.model.network.get_parameters_recipe():
            for p in p_dict['params']:
                p.requires_grad = activate_recipe
        for p in self.model.network.get_parameters_image():
            p.requires_grad = activate_image

    def _auto_fixed_fine_tune(self):
        if self.current_optimizer_name == 'recipe' and self.epoch == self.switch_epoch:
            self.current_optimizer_name = 'recipe&image'
            self._activate_model()
            Logger()('Switched to optimizer '+self.current_optimizer_name)

        Logger().log_value('optimizer.is_optimizer_recipe&image',
                           int(self.current_optimizer_name == 'recipe&image'),
                           should_print=False)
        self.epoch += 1
