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
    class_name = 'pizza'
    dir_exp = '/home/cadene/doc/bootstrap.pytorch/logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract', split)
    dir_extract_mean = os.path.join(dir_exp, 'extract_mean_features', split)
    dir_img = os.path.join(dir_extract, 'image')
    dir_rcp = os.path.join(dir_extract, 'recipe')
    path_model_ckpt = os.path.join(dir_exp, 'ckpt_best_val_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar')

    is_mean = True    
    ingrs_list = ['fresh_strawberries']#['avocado']

    
    Options(path_opts)
    Options()['misc']['seed'] = 2
    utils.set_random_seed(Options()['misc']['seed'])

    dataset = factory(split)

    Logger()('Load model...')
    model = model_factory()
    model_state = torch.load(path_model_ckpt)
    model.load_state_dict(model_state)
    model.eval()

    if not os.path.isdir(dir_extract):
        os.system('mkdir -p '+dir_rcp)
        os.system('mkdir -p '+dir_img)

        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            batch = dataset.items_tf()([item])

            if model.is_cuda:
                batch = model.cuda_tf()(batch)

            is_volatile = (model.mode not in ['train', 'trainval'])
            batch = model.variable_tf(volatile=is_volatile)(batch)

            out = model.network(batch)

            path_image = os.path.join(dir_img, '{}.pth'.format(i))
            path_recipe = os.path.join(dir_rcp, '{}.pth'.format(i))
            torch.save(out['image_embedding'][0].data.cpu(), path_image)
            torch.save(out['recipe_embedding'][0].data.cpu(), path_recipe)



    # b = dataset.make_batch_loader().__iter__().__next__()
    # import ipdb; ipdb.set_trace()
    
    ingrs = torch.LongTensor(1, len(ingrs_list))
    for i, ingr_name in enumerate(ingrs_list):
        ingrs[0, i] = dataset.recipes_dataset.ingrname_to_ingrid[ingr_name]

    input_ = {
        'recipe': {
            'ingrs': {
                'data': Variable(ingrs.cuda(), requires_grad=False),
                'lengths': [ingrs.size(1)]
            },
            'instrs': {
                'data': Variable(torch.FloatTensor(1, 1, 1024).fill_(0).cuda(), requires_grad=False),
                'lengths': [1]
            }
        }
    }

    #emb = network.recipe_embedding.forward_ingrs(input_['recipe']['ingrs'])
    #list_idx = torch.randperm(len(dataset))

    indices_by_class = dataset._make_indices_by_class()

    #import ipdb; ipdb.set_trace()

    class_id = dataset.cname_to_cid[class_name]
    list_idx = torch.Tensor(indices_by_class[class_id])
    rand_idx = torch.randperm(list_idx.size(0))
    list_idx = list_idx[rand_idx]
    list_idx = list_idx.view(-1).int()

    nb_points = list_idx.size(0)

    dir_visu = os.path.join(dir_exp, 'visu', 'ingrs_to_image_nb_points:{}_class:{}_instrs:{}_mean:{}_v2'.format(nb_points, class_name, '-'.join(ingrs_list), is_mean))
    os.system('mkdir -p '+dir_visu)

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

    Logger()('Load mean embeddings')

    path_ingrs = os.path.join(dir_extract_mean, 'ingrs.pth')
    path_instrs = os.path.join(dir_extract_mean, 'instrs.pth')

    mean_ingrs = torch.load(path_ingrs)
    mean_instrs = torch.load(path_instrs)

    Logger()('Forward ingredient...')
    #ingr_emb = model.network.recipe_embedding(input_['recipe'])
    ingr_emb = model.network.recipe_embedding.forward_one_ingr(
        input_['recipe']['ingrs'],
        emb_instrs=mean_instrs.unsqueeze(0))

    ingr_emb = ingr_emb.data.cpu()
    ingr_emb = ingr_emb.expand_as(img_embs)

    Logger()('Fast distance...')
    dist = fast_distance(img_embs, ingr_emb)[:, 0]

    sorted_ids = np.argsort(dist.numpy())

    Logger()('Load/save images in {}...'.format(dir_visu))
    for i in range(20):
        idx = int(sorted_ids[i])
        item_id = list_idx[idx]
        item = dataset[item_id]
        # path_img_from = item['image']['path']
        # ingrs = [ingr.replace('/', '\'') for ingr in item['recipe']['ingrs']['interim']]
        # cname = item['recipe']['class_name']
        # path_img_to = os.path.join(dir_visu, 'image_top:{}_ingrs:{}_cname:{}.png'.format(i+1, '-'.join(ingrs), cname))
        # img = Image.open(path_img_from)
        # img.save(path_img_to)
        #os.system('cp {} {}'.format(path_img_from, path_img_to))

        write_img_rcp(dir_visu, item, top=i, begin_with='nn')

    Logger()('End')

def write_img_rcp(dir_visu, item, top=1, begin_with=''):
    dir_visu = os.path.join(dir_visu, begin_with+'_top{}_class:{}_item:{}'.format(top, item['recipe']['class_name'].replace(' ', '_'), item['index']))
    path_rcp = dir_visu + '_rcp.txt'
    path_img = dir_visu + '_img.png'
    #os.system('mkdir -p '+dir_fig_i)

    s = [item['recipe']['layer1']['title']]
    s += ['\nIngredients raw']
    s += [d['text'] for d in item['recipe']['layer1']['ingredients']]
    s += ['\nIngredients interim']
    s += ['{}: {}'.format(item['recipe']['ingrs']['data'][idx], d) for idx, d in enumerate(item['recipe']['ingrs']['interim'])]
    s += ['\nInstructions raw']
    s += [d['text'] for d in item['recipe']['layer1']['instructions']]

    with open(path_rcp, 'w') as f:
        f.write('\n'.join(s))

    path_img_from = item['image']['path']
    img = Image.open(path_img_from)
    img.save(path_img)


    Logger()('End')



    
def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()