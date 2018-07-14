import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from . import utils

class Trijoint(nn.Module):

    def __init__(self, opt, with_classif=False, engine=None, mode='train'):
        super(Trijoint, self).__init__()
        self.mode = mode
        self.with_classif = with_classif
        self.engine = engine
        # Attributs to process 1000*10 matchs
        # for the retrieval evaluation procedure
        self.nb_bags_retrieval = opt['nb_bags']
        self.nb_matchs_per_bag = opt['nb_matchs_per_bag']
        self.nb_matchs_expected = self.nb_bags_retrieval * self.nb_matchs_per_bag
        self.nb_matchs_saved = 0

        if opt.get('keep_background', False):
            self.ignore_index = None
        else:
            self.ignore_index = 0

        self.identifiers = {'image': [], 'recipe': []}

        if engine and self.mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_metrics)

    def forward(self, cri_out, net_out, batch):
        out = {}
        if self.with_classif:
            # Accuracy
            [out['acc_image']] = utils.accuracy(net_out['image_classif'].detach().cpu(),
                                             batch['image']['class_id'].detach().squeeze().cpu(),
                                             topk=(1,),
                                             ignore_index=self.ignore_index)
            [out['acc_recipe']] = utils.accuracy(net_out['recipe_classif'].detach().cpu(),
                                              batch['recipe']['class_id'].detach().squeeze().cpu(),
                                              topk=(1,),
                                              ignore_index=self.ignore_index)
        if self.engine and self.mode == 'eval':
            # Retrieval
            batch_size = len(batch['image']['index'])
            for i in range(batch_size):
                if self.nb_matchs_saved == self.nb_matchs_expected:
                    continue
                if batch['match'].data[i][0] == -1:
                    continue

                identifier = '{}_img_{}'.format(self.split, batch['image']['index'][i])
                utils.save_activation(identifier, net_out['image_embedding'][i].detach().cpu())
                self.identifiers['image'].append(identifier)

                identifier = '{}_rcp_{}'.format(self.split, batch['recipe']['index'][i])
                utils.save_activation(identifier, net_out['recipe_embedding'][i].detach().cpu())
                self.identifiers['recipe'].append(identifier)

                self.nb_matchs_saved += 1   
        
        return out  

    def calculate_metrics(self):
        final_nb_bags = math.floor(self.nb_matchs_saved / self.nb_matchs_per_bag)
        final_matchs_left = self.nb_matchs_saved % self.nb_matchs_per_bag

        if final_nb_bags < self.nb_bags_retrieval:
            log_level = Logger.ERROR if self.split == 'test' else Logger.WARNING
            Logger().log_message('Insufficient matchs ({} saved), {} bags instead of {}'.format(
                self.nb_matchs_saved, final_nb_bags, self.nb_bags_retrieval), log_level=log_level)

        Logger().log_message('Computing retrieval ranking for {} x {} matchs'.format(final_nb_bags,
                                                                                     self.nb_matchs_per_bag))
        list_med_im2recipe = []
        list_med_recipe2im = []
        list_recall_at_1_im2recipe = []
        list_recall_at_5_im2recipe = []
        list_recall_at_10_im2recipe = []
        list_recall_at_1_recipe2im = []
        list_recall_at_5_recipe2im = []
        list_recall_at_10_recipe2im = []

        for i in range(final_nb_bags):
            nb_identifiers_image = self.nb_matchs_per_bag
            nb_identifiers_recipe = self.nb_matchs_per_bag

            distances = np.zeros((nb_identifiers_image, nb_identifiers_recipe), dtype=float)

            # load
            im_matrix = None
            rc_matrix = None
            for j in range(self.nb_matchs_per_bag):
                index = j + i * self.nb_matchs_per_bag

                identifier_image = self.identifiers['image'][index]
                activation_image = utils.load_activation(identifier_image)
                if im_matrix is None:
                    im_matrix = torch.zeros(nb_identifiers_image, activation_image.size(0))
                im_matrix[j] = activation_image

                identifier_recipe = self.identifiers['recipe'][index]
                activation_recipe = utils.load_activation(identifier_recipe)
                if rc_matrix is None:
                    rc_matrix = torch.zeros(nb_identifiers_recipe, activation_recipe.size(0))
                rc_matrix[j] = activation_recipe

            #im_matrix = im_matrix.cuda()
            #rc_matrix = rc_matrix.cuda()

            distances = fast_distance(im_matrix, rc_matrix)
            #distances[i][j] = torch.dist(activation_image.data, activation_recipe.data, p=2)

            im2recipe = np.argsort(distances.numpy(), axis=0)
            recipe2im = np.argsort(distances.numpy(), axis=1)
            
            recall_at_1_recipe2im = 0
            recall_at_5_recipe2im = 0
            recall_at_10_recipe2im = 0
            recall_at_1_im2recipe = 0
            recall_at_5_im2recipe = 0
            recall_at_10_im2recipe = 0
            med_rank_im2recipe = []
            med_rank_recipe2im = []

            for i in range(nb_identifiers_image):
                pos_im2recipe = im2recipe[:,i].tolist().index(i)
                pos_recipe2im = recipe2im[i,:].tolist().index(i)

                if pos_im2recipe == 0:
                    recall_at_1_im2recipe += 1
                if pos_im2recipe <= 4:
                    recall_at_5_im2recipe += 1
                if pos_im2recipe <= 9:
                    recall_at_10_im2recipe += 1

                if pos_recipe2im == 0:
                    recall_at_1_recipe2im += 1
                if pos_recipe2im <= 4:
                    recall_at_5_recipe2im += 1
                if pos_recipe2im <= 9:
                    recall_at_10_recipe2im += 1

                med_rank_im2recipe.append(pos_im2recipe)
                med_rank_recipe2im.append(pos_recipe2im)
            
            list_med_im2recipe.append(np.median(med_rank_im2recipe))
            list_med_recipe2im.append(np.median(med_rank_recipe2im))
            list_recall_at_1_im2recipe.append(recall_at_1_im2recipe / nb_identifiers_image)
            list_recall_at_5_im2recipe.append(recall_at_5_im2recipe / nb_identifiers_image)
            list_recall_at_10_im2recipe.append(recall_at_10_im2recipe / nb_identifiers_image)
            list_recall_at_1_recipe2im.append(recall_at_1_recipe2im / nb_identifiers_image)
            list_recall_at_5_recipe2im.append(recall_at_5_recipe2im / nb_identifiers_image)
            list_recall_at_10_recipe2im.append(recall_at_10_recipe2im / nb_identifiers_image)

        out = {}
        out['med_im2recipe_mean'] = np.mean(list_med_im2recipe)
        out['med_recipe2im_mean'] = np.mean(list_med_recipe2im)
        out['recall_at_1_im2recipe_mean'] = np.mean(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_mean'] = np.mean(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_mean'] = np.mean(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_mean'] = np.mean(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_mean'] = np.mean(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_mean'] = np.mean(list_recall_at_10_recipe2im)

        out['med_im2recipe_std'] = np.std(list_med_im2recipe)
        out['med_recipe2im_std'] = np.std(list_med_recipe2im)
        out['recall_at_1_im2recipe_std'] = np.std(list_recall_at_1_im2recipe)
        out['recall_at_5_im2recipe_std'] = np.std(list_recall_at_5_im2recipe)
        out['recall_at_10_im2recipe_std'] = np.std(list_recall_at_10_im2recipe)
        out['recall_at_1_recipe2im_std'] = np.std(list_recall_at_1_recipe2im)
        out['recall_at_5_recipe2im_std'] = np.std(list_recall_at_5_recipe2im)
        out['recall_at_10_recipe2im_std'] = np.std(list_recall_at_10_recipe2im)

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode,key), float(value), should_print=True)

        #self.outputs_epoch['total_samples'] = nb_identifiers_image

        for identifier_image in self.identifiers['image']:
            utils.delete_activation(identifier_image)

        for identifier_recipe in self.identifiers['recipe']:
            utils.delete_activation(identifier_recipe)

        self.identifiers = {'image': [], 'recipe': []}
        self.nb_matchs_saved = 0

# mAP ?
# ConfusionMatrix ?

def fast_distance(A,B):
    # A and B must have norm 1 for this to work for the ranking
    return torch.mm(A,B.t()) * -1

def euclidean_distance_fast(A,B):
    n = A.size(0)
    ZA = (A * A).sum(1)
    ZB = (B * B).sum(1)

    ZA = ZA.expand(n,n)
    ZB = ZB.expand(n,n).t()

    D = torch.mm(B, A.t())
    D.mul_(-2)
    D.add_(ZA).add_(ZB)
    D.sqrt_()
    D.t_()
    return D

def euclidean_distance_slow(A,B):
    n = A.size(0)
    D = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            D[i,j] = torch.dist(A[i], B[j])
    return D
