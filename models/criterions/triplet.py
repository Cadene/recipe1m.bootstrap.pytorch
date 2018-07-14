import torch
import numpy as np
import torch.nn as nn
import scipy.linalg as la
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class Triplet(nn.Module):

    def __init__(self, opt, nb_classes, dim_emb, engine=None):
        self.alpha = opt['retrieval_strategy']['margin']
        self.sampling = opt['retrieval_strategy']['sampling']
        self.nb_samples = opt['retrieval_strategy'].get('nb_samples', 1)
        self.substrategy = opt['retrieval_strategy'].get('substrategy', ['IRR'])
        self.aggregation = opt['retrieval_strategy'].get('aggregation', 'mean')
        self.id_background = opt['retrieval_strategy'].get('id_background', 0)
        self.nb_classes = nb_classes
        self.dim_emb = dim_emb
        
        if 'substrategy_weights' in opt['retrieval_strategy']:
            self.substrategy_weights = opt['retrieval_strategy']['substrategy_weights']
            if len(self.substrategy) > len(self.substrategy_weights):
                Logger()('Incorrect number of items in substrategy_weights (expected {}, got {})'.format(
                    len(self.substrategy), len(self.substrategy_weights)), Logger.ERROR)
            elif len(self.substrategy) < len(self.substrategy_weights):
                Logger()('Higher number of items in substrategy_weights than expected ({}, got {}). Discarding exceeding values'.format(
                    len(self.substrategy), len(self.substrategy_weights)), Logger.WARNING)
        else:
            Logger()('No substrategy_weights provided, automatically setting all items to 1', Logger.WARNING)
            self.substrategy_weights = [1.0] * len(self.substrategy)
        
        if engine:
            engine.register_hook('train_on_end_epoch', self.reset_barycenters)

    def calculate_cost(self, cost, enable_naive=True):
        if self.sampling == 'max_negative':
            ans,_ = torch.sort(cost, dim=1, descending=True)
        elif self.sampling == 'semi_hard':
            noalpha = cost - self.alpha
            mask = (noalpha <= 0)
            noalpha.masked_scatter_(mask, noalpha.max().expand_as(mask))
            ans, __argmax = torch.sort(noalpha, dim=1, descending=True)
            ans += self.alpha
        elif self.sampling == 'prob_negative':
            indexes = torch.multinomial(cost, cost.size(1))
            ans = torch.gather(cost, 1, indexes.detach())
        elif self.sampling == 'random':
            if enable_naive:
                Logger()('Random triplet strategy is outdated and does not work with non-square matrices :(', Logger.ERROR)
                indexes = la.hankel(np.roll(np.arange(cost.size(0)),-1), np.arange(cost.size(1))) # anti-circular matrix
                indexes = cost.data.new(indexes.tolist()).long()
                ans = torch.gather(cost, 1, indexes.detach())
            else:
                Logger()('Random triplet strategy not allowed with this configuration', Logger.ERROR)
        else:
            Logger()('Unknown substrategy {}.'.format(self.sampling), Logger.ERROR)
            
        return ans[:,:self.nb_samples]

    def add_cost(self, name, cost, bad_pairs, losses):
        invalid_pairs = (cost == 0).float().sum()
        bad_pairs['bad_pairs_{}'.format(name)] = invalid_pairs / cost.numel()
        if self.aggregation == 'mean':
            losses['loss_{}'.format(name)] = cost.mean() * self.substrategy_weights[self.substrategy.index(name)]
        elif self.aggregation == 'valid':
            valid_pairs = cost.numel() - invalid_pairs
            losses['loss_{}'.format(name)] = cost.sum() * self.substrategy_weights[self.substrategy.index(name)] / valid_pairs
        else:
            Logger()('Unknown aggregation strategy {}.'.format(self.aggregation), Logger.ERROR)

    def reset_barycenters(self, force=True, base_variable=None):
        if len(set(['RBB', 'IBB']).intersection(self.substrategy)) > 0:
            if not hasattr(self, 'barycenters') or force:
                if base_variable is None:
                    base_variable = self.barycenters
                self.barycenters = base_variable.data.new(self.nb_classes, self.dim_emb)
                self.barycenters[:,:] = 0
                self.counters = base_variable.data.new(self.nb_classes)
                self.counters[:] = 0

    def semantic_unimodal(self, distances, class1):
        return self.semantic_multimodal(distances, class1, class1)

    def semantic_multimodal(self, distances, class1, class2, erase_diagonal=True):
        class1_matrix = class1.squeeze(1).repeat(class1.size(0), 1)
        class2_matrix = class2.squeeze(1).repeat(class2.size(0), 1).t()
        matrix_mask = ((class1_matrix != 0) + (class2_matrix != 0)) == 2

        same_class = torch.eq(class1_matrix, class2_matrix)
        anti_class = same_class.clone()

        anti_class = anti_class == 0 # get the dissimilar classes
        if erase_diagonal:
            same_class[range(same_class.size(0)),range(same_class.size(1))] = 0 # erase instance-instance pairs
        new_dimension = matrix_mask.int().sum(1).max().item()
        same_class = torch.masked_select(same_class, matrix_mask).view(new_dimension, new_dimension)
        anti_class = torch.masked_select(anti_class, matrix_mask).view(new_dimension, new_dimension)
        mdistances = torch.masked_select(distances, matrix_mask).view(new_dimension, new_dimension)

        same_class[same_class.cumsum(dim=1) > 1] = 0 # erasing extra positives
        pos_samples = torch.masked_select(mdistances, same_class) # only the first one
        min_neg_samples = anti_class.int().sum(1).min().item() # selecting max negatives possible
        anti_class[anti_class.cumsum(dim=1) > min_neg_samples] = 0 # erasing extra negatives
        neg_samples = torch.masked_select(mdistances, anti_class).view(new_dimension, min_neg_samples)

        cost = pos_samples.unsqueeze(1) - neg_samples + self.alpha
        cost[cost < 0] = 0 # hinge
        return cost

    def __call__(self, input1, input2, target, class1, class2):
        bad_pairs = {}
        losses = {}

        # Detect and treat unbalanced batch
        size1 = class1.size(0)
        size2 = class2.size(0)
        if size1 > size2:
            exceeding_input = input1[size2:,:] # Set exceeding samples apart
            exceeding_class = class1[size2:,:] # Set exceeding samples apart
            exceeding_type = 1
            class1 = class1[:size2] # Remove exceeding samples
            input1 = input1[:size2,:] # Remove exceeding samples
            size1 = class1.size(0)
            Logger()('Size of input1 automatically reduced to balance batch (from {} to {})'.format(size1, class1.size(0)))
            if target.size(0) > size1:
                target = target[:size1]
        elif size2 > size1:
            exceeding_input = input2[size1:,:] # Set exceeding samples apart
            exceeding_class = class2[size1:,:] # Set exceeding samples apart
            exceeding_type = 2
            class2 = class2[:size1] # Remove exceeding samples
            input2 = input2[:size1,:] # Remove exceeding samples
            size2 = class2.size(0)
            Logger()('Size of input2 automatically reduced to balance batch (from {} to {})'.format(size2, class2.size(0)))
            if target.size(0) > size2:
                target = target[:size2]
        else:
            exceeding_type = 0

        # Prepare instance samples (matched pairs)
        matches = target.squeeze(1) == 1 # To support -1 or 0 as mismatch
        instance_input1 = input1[matches].view(matches.sum().int().item(), input1.size(1))
        instance_class1 = class1[matches]
        instance_input2 = input2[matches].view(matches.sum().int().item(), input2.size(1))
        instance_class2 = class2[matches]

        # Prepare semantic samples (class != 0)
        valid_input1 = class1.squeeze(1) != 0
        valid_input2 = class2.squeeze(1) != 0
        semantic_input1 = input1[valid_input1].view(valid_input1.sum().int().item(), input1.size(1))
        semantic_class1 = class1[valid_input1]
        semantic_input2 = input2[valid_input2].view(valid_input2.sum().int().item(), input2.size(1))
        semantic_class2 = class2[valid_input2]

        # Augmented semantic samples (unmatched and class != 0)
        extra_input1 = (matches == 0) + valid_input1 == 2
        extra_input2 = (matches == 0) + valid_input2 == 2
        augmented_input1 = input1[extra_input1].view(extra_input1.sum().int().item(), input1.size(1))
        augmented_class1 = class1[extra_input1]
        augmented_input2 = input2[extra_input2].view(extra_input2.sum().int().item(), input2.size(1))
        augmented_class2 = class2[extra_input2]
        
        # Instance-based triplets
        if len(set(['IRR', 'RII', 'IRI', 'RIR', 'LIFT']).intersection(self.substrategy)) > 0:
            distances = self.dist(instance_input1, instance_input2)
            if 'IRR' in self.substrategy:
                cost = distances.diag().unsqueeze(1) - distances + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('IRR', self.calculate_cost(cost), bad_pairs, losses)
            if 'RII' in self.substrategy:
                cost = distances.diag().unsqueeze(0) - distances + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('RII', self.calculate_cost(cost.t()), bad_pairs, losses)
            if 'IRI' in self.substrategy:
                distances_image = self.dist(instance_input1, instance_input1)
                cost = distances.diag().unsqueeze(1) - distances_image + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('IRI', self.calculate_cost(cost), bad_pairs, losses)
            if 'RIR' in self.substrategy:
                distances_recipe = self.dist(instance_input2, instance_input2)
                cost = distances.diag().unsqueeze(0) - distances_recipe + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('RIR', self.calculate_cost(cost), bad_pairs, losses)
            # Lifted, instance-based triplet
            if 'LIFT' in self.substrategy:
                distances_mexp = (self.alpha - distances).exp()
                sum0 = distances_mexp.sum(0)
                sum1 = distances_mexp.sum(1)
                negdiag = torch.log(sum0 + sum1 - 2*distances_mexp.diag()) # see equation 4 on the paper, this is the left side : https://arxiv.org/pdf/1511.06452.pdf
                cost = distances.diag() + negdiag
                cost[cost < 0] = 0 # hinge
                cost = cost.pow(2).sum() / 2*distances.diag().numel()
                self.add_cost('LIFT', cost, bad_pairs, losses)
            
        # Semantic-based triplets
        if len(set(['SIRR', 'SRII']).intersection(self.substrategy)) > 0:
            distances = self.dist(semantic_input1, semantic_input2)
            if 'SIRR' in self.substrategy:
                cost = self.semantic_multimodal(distances, semantic_class1, semantic_class2)
                self.add_cost('SIRR', self.calculate_cost(cost), bad_pairs, losses)

            if 'SRII' in self.substrategy:
                cost = self.semantic_multimodal(distances.t(), semantic_class2, semantic_class1)
                self.add_cost('SRII', self.calculate_cost(cost), bad_pairs, losses)

        if 'SIII' in self.substrategy:
            cost = self.semantic_unimodal(self.dist(semantic_input1, semantic_input1), semantic_class1)
            self.add_cost('SIII', self.calculate_cost(cost), bad_pairs, losses)

        if 'SRRR' in self.substrategy:
            cost = self.semantic_unimodal(self.dist(semantic_input2, semantic_input2), semantic_class2)
            self.add_cost('SRRR', self.calculate_cost(cost), bad_pairs, losses)

        out = {}
        if len(bad_pairs.keys()) > 0:
            total_bad_pairs = input1.data.new([0])
            for key in bad_pairs.keys():
                total_bad_pairs += bad_pairs[key]
                out[key] = bad_pairs[key]
            total_bad_pairs = total_bad_pairs / len(bad_pairs.keys())
            out['bad_pairs'] = total_bad_pairs
        else:
            out['bad_pairs'] = input1.data.new([0])

        total_loss = input1.data.new([0])
        if len(losses.keys()) > 0:
            for key in losses.keys():
                total_loss += losses[key]
                out[key] = losses[key]
            out['loss'] = total_loss / len(losses.keys())
        else:
            out['loss'] = input1.data.new([0])

        return out

    def dist(self, input_1, input_2):
        input_1 = nn.functional.normalize(input_1)
        input_2 = nn.functional.normalize(input_2)
        return 1 - torch.mm(input_1, input_2.t())
