import os
import torch
import numpy as np
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from recipe1m.datasets.factory import factory
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import bootstrap.lib.utils as utils

def main():

    Logger('.')

    split = 'train'
    dir_exp = '/home/cadene/doc/bootstrap.pytorch/logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract_count', split)
    path_ingrs_count = os.path.join(dir_extract, 'ingrs.pth')

    Options(path_opts)
    utils.set_random_seed(Options()['misc']['seed'])

    dataset = factory(split)

    if not os.path.isfile(path_ingrs_count):
        ingrs_count = {}
        os.system('mkdir -p '+dir_extract)

        for i in tqdm(range(len(dataset.recipes_dataset))):
            item = dataset.recipes_dataset[i]
            for ingr in item['ingrs']['interim']:
                if ingr not in ingrs_count:
                    ingrs_count[ingr] = 1
                else:
                    ingrs_count[ingr] += 1

        torch.save(ingrs_count, path_ingrs_count)
    else:
        ingrs_count = torch.load(path_ingrs_count)

    import ipdb; ipdb.set_trace()
    sort = sorted(ingrs_count, key=ingrs_count.get)
    import ipdb; ipdb.set_trace()

    Logger()('End')


# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()