
import argparse
import os
import scipy.io as sio
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)

import torch
import torchvision.transforms as viztransforms
import bootstrap.datasets.transforms as transforms
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from PIL import Image
from scipy.misc import imsave

def stack():
    return transforms.Compose([
        transforms.ListDictsToDictLists(),
        transforms.StackTensors()
    ])

def load_image(path, crop_size=50):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_rgb = img.convert('RGB')
    img_rgb = viztransforms.Scale(crop_size)(img_rgb)
    img_rgb = viztransforms.CenterCrop(crop_size)(img_rgb)
    #img_rgb = viztransforms.ToTensor()(img_rgb)
    # img_rgb = img_rgb.transpose(0,2)
    # img_rgb = img_rgb.transpose(0,1)
    # img_rgb = img_rgb.numpy() * 255
    return img_rgb


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--perplexity', type=float, default=5.0, nargs='+', help='')
parser.add_argument('--exaggeration', type=float, default=12.0, nargs='+', help='')


def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

def load_embs(dir_root):
    dir_embs = os.path.join(dir_root, 'embeddings_test')
    dir_img_items = os.path.join(dir_embs, 'img')
    dir_rcp_items = os.path.join(dir_embs, 'rcp')
    path_load_img_out = os.path.join(dir_embs, 'load_img_out.pth')
    path_load_rcp_out = os.path.join(dir_embs, 'load_rcp_out.pth')

    if not os.path.isfile(path_load_img_out):
        Logger()('Loading embeddings...')
        rcp_items = []
        img_items = []
        for filename in os.listdir(dir_img_items):
            idx = int(filename.split('.')[0].split('_')[-1])
            #
            path_item = os.path.join(dir_img_items, filename)
            img_item = torch.load(path_item)
            img_item['idx'] = idx
            img_items.append(img_item)
            #
            path_item = path_item.replace('img', 'rcp')
            rcp_item = torch.load(path_item)
            rcp_item['idx'] = idx
            rcp_items.append(rcp_item)

        Logger()('Stacking image items...')
        X_img = stack()(img_items)
        torch.save(X_img, path_load_img_out)

        Logger()('Stacking rcpipe items...')
        X_rcp = stack()(rcp_items)
        torch.save(X_rcp, path_load_rcp_out)
    else:
        X_img = torch.load(path_load_img_out)
        X_rcp = torch.load(path_load_rcp_out)
    return X_img, X_rcp

def filter_img(X, classes, nb_points):
    X_new = {
        'path':[],
        'class_name':[],
        'index':[],
        'data':[],
        'class_id':[],
        'idx': []
    }
    idx = 0
    for c in classes:
        for i in range(len(X['index'])):
            if c == X['class_name'][i]:
                X_new['path'].append(X['path'][i])
                X_new['class_name'].append(X['class_name'][i])
                X_new['index'].append(X['index'][i])
                X_new['data'].append(X['data'][i])
                X_new['class_id'].append(X['class_id'][i])
                X_new['idx'].append(X['idx'][i])
                idx += 1
                if idx >= nb_points:
                    break
        # if idx < nb_points:
        #     Logger()('Warning: classe {} has {} items'.format(c, idx))
    X_new['data'] = transforms.StackTensors()(X_new['data'])
    return X_new 

def filter_rcp(X, classes, nb_points):
    X_new = {
        'url':[],
        'ingredients':[],
        'instructions':[],
        'title':[],
        'class_name':[],
        'index':[],
        'data':[],
        'class_id':[],
        'idx': []
    }
    idx = 0
    for c in classes:
        for i in range(len(X['index'])):
            if c == X['class_name'][i]:
                X_new['url'].append(X['url'][i])
                X_new['ingredients'].append(X['ingredients'][i])
                X_new['instructions'].append(X['instructions'][i])
                X_new['title'].append(X['title'][i])
                X_new['class_name'].append(X['class_name'][i])
                X_new['index'].append(X['index'][i])
                X_new['data'].append(X['data'][i])
                X_new['class_id'].append(X['class_id'][i])
                X_new['idx'].append(X['idx'][i])
                idx += 1
                if idx >= nb_points:
                    break
        # if idx < nb_points:
        #     Logger()('Warning: classe {} has {} items'.format(c, idx))
    X_new['data'] = transforms.StackTensors()(X_new['data'])
    return X_new 


def main():
    global args 
    args = parser.parse_args()

    Logger('.')
    Logger()('Begin')

    classes = ['pizza', 'pork chops', 'cupcake', 'hamburger', 'green beans']
    classes = ['ice cream']
    nb_points = 10000

    
    dir_root_triplet = '/home/carvalho/experiments/im2recipe.pytorch/logs/lmdb/2017_10_06_08_10_47_631_anm_IRR_RII_80epochs'

    dir_fig_triplet = os.path.join(dir_root_triplet,'visualization_top5')
    os.system('rm -rf '+dir_fig_triplet)
    os.system('mkdir -p '+dir_fig_triplet)

    X_img_triplet, X_rcp_triplet = load_embs(dir_root_triplet)

    classes = list(set(X_img_triplet['class_name']))
    new_classes = []
    for c in classes:
        if c != 'background':
            new_classes.append(c)
    classes=new_classes
    # nb_items_by_class = {class_name:0 for class_name in all_classes}
    # for i in range(len(X_img_triplet['index'])):
    #     class_name = X_img_triplet['class_name'][i]
    #     nb_items_by_class[class_name] += 1

    # for key, value in sorted(nb_items_by_class.items(), key=lambda item: item[1]):
    #     print("{}: {}".format(key, value))
    # import ipdb;ipdb.set_trace()

    X_img_triplet = filter_img(X_img_triplet, classes, nb_points)
    X_rcp_triplet = filter_rcp(X_rcp_triplet, classes, nb_points)

    distances_triplet = fast_distance(X_img_triplet['data'], X_rcp_triplet['data'])

    im2recipe_triplet = np.argsort(distances_triplet.numpy(), axis=0)
    recipe2im_triplet = np.argsort(distances_triplet.numpy(), axis=1) 





    dir_root_tri_sem = '/home/carvalho/experiments/im2recipe.pytorch/logs/lmdb/2017_10_10_23_54_31_517_anm_IRR1.0_RII1.0_SIRR0.1_SRII0.1_80epochs'

    dir_fig_tri_sem = os.path.join(dir_root_tri_sem,'visualization_top5')
    os.system('rm -rf '+dir_fig_tri_sem)
    os.system('mkdir -p '+dir_fig_tri_sem)

    X_img_tri_sem, X_rcp_tri_sem = load_embs(dir_root_tri_sem)
    X_img_tri_sem = filter_img(X_img_tri_sem, classes, nb_points)
    X_rcp_tri_sem = filter_rcp(X_rcp_tri_sem, classes, nb_points)

    distances_tri_sem = fast_distance(X_img_tri_sem['data'], X_rcp_tri_sem['data'])

    im2recipe_tri_sem = np.argsort(distances_tri_sem.numpy(), axis=0)
    recipe2im_tri_sem = np.argsort(distances_tri_sem.numpy(), axis=1) 


    for i in range(500000):


        

        # triplet_sem nb_same class > triplet nb_same_claa


        # if triplet less good

        # if top5 tri_sem has 5 classes

        # if top5 triplet has less than 3 classes

        # if tri_sem first rank 

        pos_tri_sem = None
        for j in range(5):
            id_img = recipe2im_tri_sem[i,j]
            if i == id_img:
                pos_tri_sem = j

        if pos_tri_sem is None:
            continue

        pos_triplet = None
        for j in range(5):
            id_img = recipe2im_triplet[i,j]
            if i == id_img:
                pos_triplet = j

        if pos_triplet is None:
            continue

        # if triplet_sem better than triplet
        if pos_tri_sem > pos_triplet:
            continue

        if pos_tri_sem != 0:
            continue

        class_name = X_rcp_tri_sem['class_name'][i]
        nb_same_class_tri_sem = 0
        for j in range(5):
            id_img = recipe2im_tri_sem[i,j]
            if class_name == X_img_tri_sem['class_name'][id_img]:
                nb_same_class_tri_sem += 1

        class_name = X_rcp_triplet['class_name'][i]
        nb_same_class_triplet = 0
        for j in range(5):
            id_img = recipe2im_triplet[i,j]
            if class_name == X_img_triplet['class_name'][id_img]:
                nb_same_class_triplet += 1

        if nb_same_class_tri_sem <= nb_same_class_triplet:
            continue

        if nb_same_class_tri_sem != 5:
            continue


        # if recipe2im_tri_sem[i,0] != i:
        #     continue

        # if recipe2im_triplet[i,0] == i:
        #     continue

        # class_name = X_rcp_tri_sem['class_name'][i]
        # nb_same_class = 0
        # for j in range(5):
        #     id_img = recipe2im_triplet[i,j]
        #     if class_name == X_img_triplet['class_name'][id_img]:
        #         nb_same_class += 1

        # if nb_same_class == 5:
        #     continue

        Logger()('{} found'.format(i))
        print(nb_same_class_tri_sem, nb_same_class_triplet)
        print(pos_tri_sem, pos_triplet)

        write_img_rcp(dir_fig_tri_sem, i, X_img_tri_sem, X_rcp_tri_sem, recipe2im_tri_sem)
        write_img_rcp(dir_fig_triplet, i, X_img_triplet, X_rcp_triplet, recipe2im_triplet)


    Logger()('End')


def write_img_rcp(dir_fig, i, X_img, X_rcp, recipe2im):
    dir_fig_i = os.path.join(dir_fig, 'fig_{}_{}'.format(i, X_rcp['class_name'][i].replace(' ', '_')))
    os.system('mkdir -p '+dir_fig_i)

    s = [X_rcp['title'][i]]
    s += [d['text'] for d in X_rcp['ingredients'][i]]
    s += [d['text'] for d in X_rcp['instructions'][i]]

    path_rcp = os.path.join(dir_fig_i, 'rcp.txt')
    with open(path_rcp, 'w') as f:
        f.write('\n'.join(s))

    for j in range(5):
        id_img = recipe2im[i,j]
        path_img_load = X_img['path'][id_img]
        class_name = X_img['class_name'][id_img]
        class_name = class_name.replace(' ', '-')
        if id_img == i:
            path_img_save = os.path.join(dir_fig_i, 'img_{}_{}_found.png'.format(j, class_name))
        else:
            path_img_save = os.path.join(dir_fig_i, 'img_{}_{}.png'.format(j, class_name))
        I = load_image(path_img_load, crop_size=500)
        I.save(path_img_save)      

if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except:
    #     try:
    #         Logger()(traceback.format_exc(), Logger.ERROR)
    #     except:
    #         pass
    #     pass