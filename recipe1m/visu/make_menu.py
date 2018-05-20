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

    #classes = ['hamburger']
    #nb_points = 
    split = 'test'
    class_name = None#'potato salad'
    modality_to_modality = 'recipe_to_image'
    dir_exp = '/home/cadene/doc/bootstrap.pytorch/logs/recipe1m/trijoint/2017-12-14-15-04-51'
    path_opts = os.path.join(dir_exp, 'options.yaml')
    dir_extract = os.path.join(dir_exp, 'extract', split)
    dir_extract_mean = os.path.join(dir_exp, 'extract_mean_features', split)
    dir_img = os.path.join(dir_extract, 'image')
    dir_rcp = os.path.join(dir_extract, 'recipe')
    path_model_ckpt = os.path.join(dir_exp, 'ckpt_best_val_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar')

    

    #is_mean = True    
    #ingrs_list = ['carotte', 'salad', 'tomato']#['avocado']

    #Options(path_opts)
    Options(path_opts)
    Options()['misc']['seed'] = 11
    utils.set_random_seed(Options()['misc']['seed'])

    chosen_item_id = 51259
    dataset = factory(split)
    if class_name:
        class_id = dataset.cname_to_cid[class_name]
        indices_by_class = dataset._make_indices_by_class()
        nb_points = len(indices_by_class[class_id])
        list_idx = torch.Tensor(indices_by_class[class_id])
        rand_idx = torch.randperm(list_idx.size(0))
        list_idx = list_idx[rand_idx]
        list_idx = list_idx.view(-1).int()
        dir_visu = os.path.join(dir_exp, 'visu', 'remove_ingrs_item:{}_nb_points:{}_class:{}'.format(chosen_item_id, nb_points, class_name.replace(' ', '_')))
    else:
        nb_points = 1000
        list_idx = torch.randperm(len(dataset))
        dir_visu = os.path.join(dir_exp, 'visu', 'remove_ingrs_item:{}_nb_points:{}_removed'.format(chosen_item_id, nb_points))


    # for i in range(20):
    #     item_id = list_idx[i]
    #     item = dataset[item_id]
    #     write_img_rcp(dir_visu, item, top=i)

    Logger()('Load model...')
    model = model_factory()
    model_state = torch.load(path_model_ckpt)
    model.load_state_dict(model_state)
    model.eval()

    item = dataset[chosen_item_id]

    # from tqdm import tqdm
    # ids = []
    # for i in tqdm(range(len(dataset.recipes_dataset))):
    #     item = dataset.recipes_dataset[i]#23534]
    #     if 'broccoli' in item['ingrs']['interim']:
    #         print('broccoli', i)
    #         ids.append(i)
            
    #     # if 'mushroom' in item['ingrs']['interim']:
    #     #     print('mushroom', i)
    #     #     break

    import ipdb; ipdb.set_trace()

    

    # input_ = {
    #     'recipe': {
    #         'ingrs': {
    #             'data': item['recipe']['ingrs']['data'],
    #             'lengths': item['recipe']['ingrs']['lengths']
    #         },
    #         'instrs': {
    #             'data': item['recipe']['instrs']['data'],
    #             'lengths': item['recipe']['instrs']['lengths']
    #         }
    #     }
    # }

    instrs = torch.FloatTensor(6,1024)
    instrs[0] = item['recipe']['instrs']['data'][0]
    instrs[1] = item['recipe']['instrs']['data'][1]
    instrs[2] = item['recipe']['instrs']['data'][3]
    instrs[3] = item['recipe']['instrs']['data'][4]
    instrs[4] = item['recipe']['instrs']['data'][6]
    instrs[5] = item['recipe']['instrs']['data'][7]

    ingrs = torch.LongTensor([612,585,844,3087,144,188,1])

    input_ = {
        'recipe': {
            'ingrs': {
                'data': ingrs,
                'lengths': ingrs.size(0)
            },
            'instrs': {
                'data': instrs,
                'lengths': instrs.size(0)
            }
        }
    }

    batch = dataset.items_tf()([input_])
    batch = model.prepare_batch(batch)
    out = model.network.recipe_embedding(batch['recipe'])

    # path_rcp = os.path.join(dir_rcp, '{}.pth'.format(23534))
    # rcp_emb = torch.load(path_rcp)
    

    Logger()('Load embeddings...')
    img_embs = []
    for i in range(nb_points):
        try:
            idx = list_idx[i]
        except:
            import ipdb; ipdb.set_trace()
        #idx = i
        path_img = os.path.join(dir_img, '{}.pth'.format(idx))
        if not os.path.isfile(path_img):
            Logger()('No such file: {}'.format(path_img))
            continue
        img_embs.append(torch.load(path_img))

    img_embs = torch.stack(img_embs, 0)

    Logger()('Fast distance...')

    dist = fast_distance(out.data.cpu().expand_as(img_embs), img_embs)
    dist = dist[0, :]
    sorted_ids = np.argsort(dist.numpy())

    os.system('rm -rf '+dir_visu)
    os.system('mkdir -p '+dir_visu)

    Logger()('Load/save images in {}...'.format(dir_visu))
    write_img_rcp(dir_visu, item, top=0, begin_with='query')
    for i in range(20):
        idx = int(sorted_ids[i])
        item_id = list_idx[idx]
        #item_id = idx
        item = dataset[item_id]
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

    
def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

# python -m recipe1m.visu.top5
if __name__ == '__main__':
    main()