import os
import math
import time
import torch
import datetime
from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.engines.engine import Engine

class Extract(Engine):

    def __init__(self):
        super(Extract, self).__init__()

    def eval_epoch(self, model, dataset, epoch, mode='val', logs_json=False):
        utils.set_random_seed(Options()['misc']['seed'] + epoch) # to be able to reproduce exps on reload
        Logger()('Extract model on {}set for epoch {}'.format(dataset.split, epoch))
        model.set_mode(mode)

        dir_extract = os.path.join(Options()['exp']['dir'], 'extract', mode)
        dir_recipe = os.path.join(dir_extract, 'recipe')
        dir_image = os.path.join(dir_extract, 'image')
        os.system('mkdir -p '+dir_recipe)
        os.system('mkdir -p '+dir_image)

        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }

        out_epoch = {}
        batch_loader = dataset.make_batch_loader()

        self.hook(mode+'_on_start_epoch')
        for i, batch in enumerate(batch_loader):
            self.hook(mode+'_on_start_batch')
            timer['load'] = time.time() - timer['elapsed']

            if model.is_cuda:
                batch = model.cuda_tf()(batch)

            is_volatile = (model.mode not in ['train', 'trainval'])
            batch = model.variable_tf(volatile=is_volatile)(batch)

            out = model.network(batch)
            self.hook(mode+'_on_forward')

            for j, idx in enumerate(batch['recipe']['index']):
                path_image = os.path.join(dir_image, '{}.pth'.format(idx))
                path_recipe = os.path.join(dir_recipe, '{}.pth'.format(idx))
                torch.save(out['image_embedding'][j].data.cpu(), path_image)
                torch.save(out['recipe_embedding'][j].data.cpu(), path_recipe)

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('train_batch.batch', i, should_print=False)
            Logger().log_value('train_batch.epoch', epoch, should_print=False)
            Logger().log_value('train_batch.timer.process', timer['process'], should_print=False)
            Logger().log_value('train_batch.timer.load', timer['load'], should_print=False)

            for key, value in out.items():
                if value.dim() == 1 and value.size(0) == 1:
                    if key not in out_epoch:
                        out_epoch[key] = []
                    if hasattr(value, 'data'):
                        value = value.data[0]
                    out_epoch[key].append(value)
                    Logger().log_value('train_batch.'+key, value, should_print=False)

            if i % Options()['misc']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{}  elapsed: {} | left: {}".format(' '*len(mode), 
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{}  process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
            
            timer['elapsed'] = time.time()
            self.hook(mode+'_on_end_batch')

            if Options()['misc']['debug']:
                if i > 20:
                    break

        out = {}
        for key, value in out_epoch.items():
            out[key] = sum(value)/len(value)

        Logger().log_value(mode+'_epoch.epoch', epoch, should_print=True)
        for key, value in out.items():
            Logger().log_value(mode+'_epoch.'+key, value, should_print=True)

        self.hook(mode+'_on_end_epoch')
        if logs_json:
            Logger().flush()

        return out
