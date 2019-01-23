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
    nb_points = 1000
    split = 'test'
    dir_exp = '/home/cadene/doc/bootstrap.pytorch/logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract', split)
    dir_extract_mean = os.path.join(dir_exp, 'extract_mean_features', split)
    dir_img = os.path.join(dir_extract, 'image')
    dir_rcp = os.path.join(dir_extract, 'recipe')
    path_model_ckpt = os.path.join(dir_exp, 'ckpt_best_val_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar')
    
    dir_visu = os.path.join(dir_exp, 'visu', 'mean_to_image')
    os.system('mkdir -p '+dir_visu)

    #Options(path_opts)
    Options.load_from_yaml(path_opts)
    utils.set_random_seed(Options()['misc']['seed'])

    dataset = factory(split)

    Logger()('Load model...')
    model = model_factory()
    model_state = torch.load(path_model_ckpt)
    model.load_state_dict(model_state)
    model.set_mode(split)

    #emb = network.recipe_embedding.forward_ingrs(input_['recipe']['ingrs'])
    list_idx = torch.randperm(len(dataset))

    Logger()('Load embeddings...')
    img_embs = []
    rcp_embs = []
    for i in range(nb_points):
        idx = list_idx[i]
        path_img = os.path.join(dir_img, '{}.pth'.format(idx))
        path_rcp = os.path.join(dir_rcp, '{}.pth'.format(idx))
        if not os.path.isfile(path_img):
            Logger()('No such file: {}'.format(path_img))
            continue
        if not os.path.isfile(path_rcp):
            Logger()('No such file: {}'.format(path_rcp))
            continue
        img_embs.append(torch.load(path_img))
        rcp_embs.append(torch.load(path_rcp))

    img_embs = torch.stack(img_embs, 0)
    rcp_embs = torch.stack(rcp_embs, 0)


    Logger()('Load means')
    path_ingrs = os.path.join(dir_extract_mean, 'ingrs.pth')
    path_instrs = os.path.join(dir_extract_mean, 'instrs.pth')

    mean_ingrs = torch.load(path_ingrs)
    mean_instrs = torch.load(path_instrs)

    mean_ingrs = Variable(mean_ingrs.unsqueeze(0).cuda(), requires_grad=False)
    mean_instrs = Variable(mean_instrs.unsqueeze(0).cuda(), requires_grad=False)

    Logger()('Forward ingredient...')
    ingr_emb = model.network.recipe_embedding.forward_ingrs_instrs(mean_ingrs, mean_instrs)
    ingr_emb = ingr_emb.data.cpu()
    ingr_emb = ingr_emb.expand_as(img_embs)

    Logger()('Fast distance...')
    dist = fast_distance(img_embs, ingr_emb)[:, 0]

    sorted_img_ids = np.argsort(dist.numpy())

    Logger()('Load/save images...')
    for i in range(20):
        img_id = sorted_img_ids[i]
        img_id = int(img_id)

        path_img_from = dataset[img_id]['image']['path']
        path_img_to = os.path.join(dir_visu, 'image_top_{}.png'.format(i+1))
        img = Image.open(path_img_from)
        img.save(path_img_to)
        #os.system('cp {} {}'.format(path_img_from, path_img_to))

    Logger()('End')

    
def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()