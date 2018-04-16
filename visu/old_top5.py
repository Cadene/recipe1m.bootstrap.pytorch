import os
import torch
import numpy as np
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from recipe1m.datasets.factory import factory
from recipe1m.models.networks.trijoint import Trijoint

def main():
    classes = ['pizza', 'pork chops', 'cupcake', 'hamburger', 'green beans']
    nb_points = 100
    split = 'test'
    dir_exp = 'logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract', split)
    dir_img = os.path.join(dir_extract, 'image')
    dir_rcp = os.path.join(dir_extract, 'recipe')
    path_model = os.path.join(dir_exp, 'ckpt_best_val_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar')
    
    dir_visu = os.path.join(dir_exp, 'visu', 'top5')

    #Options(path_opts)
    Options.load_from_yaml(path_opts)
    dataset = factory(split)

    network = Trijoint()
    network.eval()
    model_state = torch.load(path_model)
    network.load_state_dict(model_state['network'])

    list_idx = torch.randperm(len(dataset))

    img_embs = []
    rcp_embs = []
    for i in range(nb_points):
        idx = list_idx[i]
        path_img = os.path.join(dir_img, '{}.pth'.format(idx))
        path_rcp = os.path.join(dir_rcp, '{}.pth'.format(idx))
        img_embs.append(torch.load(path_img))
        rcp_embs.append(torch.load(path_rcp))

    img_embs = torch.stack(img_embs, 0)
    rcp_embs = torch.stack(rcp_embs, 0)

    dist = fast_distance(img_embs, rcp_embs)

    im2recipe_ids = np.argsort(dist.numpy(), axis=0)
    recipe2im_ids = np.argsort(dist.numpy(), axis=1) 

    import ipdb; ipdb.set_trace()


    
def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()