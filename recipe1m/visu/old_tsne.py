
import os
import torch
from sklearn.manifold import TSNE
import argparse
import scipy.io as sio
import numpy as np
import bootstrap.datasets.transforms as transforms
import torchvision.transforms as viztransforms
from bootstrap.lib.logger import Logger
from PIL import Image

from scipy.misc import imsave

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)



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
    img_rgb = viztransforms.ToTensor()(img_rgb)
    img_rgb = img_rgb.transpose(0,2)
    img_rgb = img_rgb.transpose(0,1)
    img_rgb = img_rgb.numpy() * 255
    return img_rgb


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--perplexity', type=float, default=5.0, nargs='+', help='')
parser.add_argument('--exaggeration', type=float, default=12.0, nargs='+', help='')

def main():
    global args 
    args = parser.parse_args()

    #dir_root = '/home/carvalho/experiments/im2recipe.pytorch/logs/lmdb/2017_10_10_23_54_31_517_anm_IRR1.0_RII1.0_SIRR0.1_SRII0.1_80epochs'
    dir_root = '/home/carvalho/experiments/im2recipe.pytorch/logs/lmdb/2017_10_06_08_10_47_631_anm_IRR_RII_80epochs'
    dir_embs = os.path.join(dir_root, 'embeddings_train')
    dir_img_items = os.path.join(dir_embs, 'img')
    dir_rcp_items = os.path.join(dir_embs, 'rcp')
    #dir_img_items = os.path.join(dir_img_embs, '2017_10_10_23_54_31_517_anm_IRR1.0_RII1.0_SIRR0.1_SRII0.1_80epochs')
    #dir_rcp_items = os.path.join(dir_rcp_embs, '2017_10_10_23_54_31_517_anm_IRR1.0_RII1.0_SIRR0.1_SRII0.1_80epochs')
    path_load_img_out = os.path.join(dir_embs, 'load_img_out.pth')
    path_load_rcp_out = os.path.join(dir_embs, 'load_rcp_out.pth')

    dir_visu = os.path.join(dir_root, 'visualizations')
    dir_fig = os.path.join(dir_visu, 'fig_train_6')
    os.system('mkdir -p '+dir_fig)

    Logger(dir_fig)
    Logger()('Begin')

    rcp_items = []
    img_items = []
    if not os.path.isfile(path_load_img_out):
        Logger()('Loading embeddings...')

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


    # all_classes = list(set(X_img['class_name']))
    # nb_items_by_class = {class_name:0 for class_name in all_classes}
    # for i in range(len(X_img['index'])):
    #     class_name = X_img['class_name'][i]
    #     nb_items_by_class[class_name] += 1

    # for key, value in sorted(nb_items_by_class.items(), key=lambda item: item[1]):
    #     print("{}: {}".format(key, value))
    # import ipdb;ipdb.set_trace()

    #print(set(X_img['class_name']))
    #return

    X_new = {
        #'path':[],
        'class_name':[],
        'index':[],
        'data':[],
        'class_id':[],
        'type':[],
        'idx': []
    }

    # tiramisu
    # bread salad
    # pork chop
    #classes = ['pork chop', 'strawberry pie', 'cheddar cheese', 'greek salad', 'curry chicken']
    #classes = ['bell pepper', 'chocolate banana', 'celery root', 'fruit salad', 'pasta sauce']
    #classes = ['chocolate banana', 'lemon pepper', 'fruit salad', 'pasta sauce']

    classes = ['pizza', 'pork chops', 'cupcake', 'hamburger', 'green beans']
    #classes = ['sweet potato', 'pizza', 'chocolate chip', 'crock pot', 'peanut butter']
    #classes = ['crock pot', 'peanut butter']
    colors = ['crimson', 'darkgreen', 'navy', 'darkorange', 'deeppink']
    class_color = {class_name: colors[i] for i, class_name in enumerate(classes)}

    Logger()(classes)

    nb_points = 80
    total_nb_points = len(classes)*nb_points*2

    def filter(X, X_new, classes, nb_points):
        for c in classes:
            idx = 0
            for i in range(len(X['index'])):
                if c == X['class_name'][i]:
                    if 'path' in X:
                        X_new['type'].append('img')
                    else:
                        X_new['type'].append('rcp')

                    X_new['class_name'].append(X['class_name'][i])
                    X_new['index'].append(X['index'][i])
                    X_new['data'].append(X['data'][i])
                    X_new['class_id'].append(X['class_id'][i])
                    X_new['idx'].append(X['idx'][i])
                    
                    idx += 1
                    if idx >= nb_points:
                        break
            if idx < nb_points:
                Logger()('Warning: classe {} has {} items'.format(c, idx))
        return X_new

    X_new = filter(X_img, X_new, classes, nb_points)
    X_new = filter(X_rcp, X_new, classes, nb_points)


    def shuffle(X):
        length = int(len(X['index'])/2)
        indexes = list(torch.randperm(length))
        X_new = {}
        for key in ['class_name', 'index', 'data', 'class_id', 'type', 'idx']:
            X_new[key] = []
            # shuffle img
            for idx in indexes:
                X_new[key].append(X[key][idx])
            # shuffle rcp
            for idx in indexes:
                X_new[key].append(X[key][idx+length])
        return X_new

    X = shuffle(X_new)
    X['data'] = transforms.StackTensors()(X['data'])
    X['data'] = X['data'].numpy()

    print(X['data'].shape)

    for perplexity in range(0,100,2):#args.perplexity:
        for exaggeration in range(1,10,2):#args.exaggeration:

            path_tsne_out = os.path.join(dir_fig, 'ckpt_perplexity,{}_exaggeration,{}.pth'.format(perplexity, exaggeration))
            if True or not os.path.isfile(path_tsne_out):

                Logger()('Calculating TSNE...')
                X_embedded = TSNE(n_components=2,
                                  perplexity=perplexity,
                                  early_exaggeration=exaggeration,
                                  learning_rate=100.0,
                                  n_iter=5000,
                                  n_iter_without_progress=300,
                                  min_grad_norm=1e-07,
                                  metric='euclidean',
                                  init='random',#'random',
                                  verbose=0,
                                  random_state=None,
                                  method='exact',#'barnes_hut',
                                  angle=0.5).fit_transform(X['data'])

                torch.save(torch.from_numpy(X_embedded), path_tsne_out)
            else:
                X_embedded = torch.load(path_tsne_out).numpy()

            Logger()('Painting...')
            # set min point to 0 and scale
            X_embedded = X_embedded - np.min(X_embedded)
            X_embedded = X_embedded / np.max(X_embedded)

            X['tsne'] = []
            for i in range(total_nb_points):
                X['tsne'].append(X_embedded[i])



            # X_img_per_class = {}
            # X_rcp_per_class = {}
            # for c in classes:
            #     X_img_per_class[c] = np.zeros((nb_points,2))
            #     X_rcp_per_class[c] = np.zeros((nb_points,2))

            #     idx_img = 0
            #     idx_rcp = 0
            #     for i in range(len(X['idx'])):
            #          if c == X['class_name'][i]:
            #             if X['type'][i] == 'img':
            #                 X_img_per_class[c][idx_img] = X_embedded[i]
            #                 idx_img += 1
            #             else:
            #                 X_rcp_per_class[c][idx_rcp] = X_embedded[i]
            #                 idx_rcp += 1

            # import ipdb; ipdb.set_trace()
            
            fig = plt.figure(figsize=(20,20))
            #ax = plt.subplot(111)
            
            for i in range(total_nb_points):
                if X['type'][i] == 'img':
                    marker = '+'
                else:
                    marker = '.'
                class_name = X['class_name'][i]
                color = class_color[class_name]
                x = X['tsne'][i][0]
                y = X['tsne'][i][1]
                plt.scatter(x, y, color=color, marker=marker, label=class_name, s=1000)

            nb_img_points = int(total_nb_points/2)
            for i in range(nb_img_points):
                class_name = X['class_name'][i]
                color = class_color[class_name]
                img_x = X['tsne'][i][0]
                img_y = X['tsne'][i][1]
                rcp_x = X['tsne'][nb_img_points+i][0]
                rcp_y = X['tsne'][nb_img_points+i][1]
                plt.plot([img_x, rcp_x], [img_y, rcp_y], '-', color=color, lw=2)

            #plt.grid(True)
            #plt.xticks([x/10 for x in range(0,11)])
            #plt.yticks([x/10 for x in range(0,11)])
            #plt.legend()

            plt.yticks([])
            plt.xticks([])

            path_fig = os.path.join(dir_fig, 'fig_perplexity,{}_exaggeration,{}.png'.format(perplexity, exaggeration))
            fig.savefig(path_fig)
            Logger()('Saved fig to '+path_fig)

            plt.show()

    Logger()('End')


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