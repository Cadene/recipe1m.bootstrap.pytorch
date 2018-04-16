import os
import torch
import numpy as np
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from recipe1m.datasets.factory import factory
from bootstrap.models.factory import factory as model_factory
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import bootstrap.lib.utils as utils

def main():

    Logger('.')

    #classes = ['pizza', 'pork chops', 'cupcake', 'hamburger', 'green beans']
    split = 'test'
    dir_exp = '/home/cadene/doc/bootstrap.pytorch/logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract_mean_features', split)
    dir_img = os.path.join(dir_extract, 'image')
    dir_rcp = os.path.join(dir_extract, 'recipe')
    path_model_ckpt = os.path.join(dir_exp, 'ckpt_best_val_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar')
    
    Options.load_from_yaml(path_opts)
    utils.set_random_seed(Options()['misc']['seed'])

    Logger()('Load dataset...')
    dataset = factory(split)

    Logger()('Load model...')
    model = model_factory()
    model_state = torch.load(path_model_ckpt)
    model.load_state_dict(model_state)
    model.set_mode(split)

    if not os.path.isdir(dir_extract):
        Logger()('Create extract_dir {}'.format(dir_extract))
        os.system('mkdir -p '+dir_extract)

        mean_ingrs = torch.zeros(model.network.recipe_embedding.dim_ingr_out*2) # bi LSTM
        mean_instrs = torch.zeros(model.network.recipe_embedding.dim_instr_out)

        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            batch = dataset.items_tf()([item])

            batch = model.prepare_batch(batch)
            out_ingrs = model.network.recipe_embedding.forward_ingrs(batch['recipe']['ingrs'])
            out_instrs = model.network.recipe_embedding.forward_instrs(batch['recipe']['instrs'])

            mean_ingrs += out_ingrs.data.cpu().squeeze(0)
            mean_instrs += out_instrs.data.cpu().squeeze(0)

        mean_ingrs /= len(dataset)
        mean_instrs /= len(dataset)

        path_ingrs = os.path.join(dir_extract, 'ingrs.pth')
        path_instrs = os.path.join(dir_extract, 'instrs.pth')

        torch.save(mean_ingrs, path_ingrs)
        torch.save(mean_instrs, path_instrs)

    Logger()('End')


# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()