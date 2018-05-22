import torch
import torch.nn as nn
from torch.autograd import Variable
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class Quadruplet(nn.Module):

    def __init__(self):
        super(Quadruplet, self).__init__()
        self.alpha = Options()['model']['criterion']['retrieval_strategy']['margin']
        self.sampling = Options()['model']['criterion']['retrieval_strategy']['sampling']

    def forward(self, image_embedding, recipe_embedding, target):
        if 'n_samples' in Options()['model']['criterion']['retrieval_strategy']:
            n_samples = Options()['model']['criterion']['retrieval_strategy']['n_samples']
        else:
            n_samples = 1
        distances = self.dist(image_embedding, recipe_embedding)
        dist_pos = distances.diag()

        if self.sampling == 'max_negative':
            weight_neg = distances.clone()
            weight_neg[range(distances.size(0)),range(distances.size(0))] = 2
            weight_neg = 2 - weight_neg # From dist to similarity, yielding + prob to closer negs
            _, indexes = torch.max(weight_neg, 0)
            dist_neg = distances[range(distances.size(0)),indexes.data.squeeze().tolist()]
        elif self.sampling == 'prob_negative':
            weight_neg = distances.clone()
            weight_neg[range(distances.size(0)),range(distances.size(0))] = 2
            weight_neg = 2 - weight_neg # From dist to similarity, yielding + prob to closer negs
            indexes = torch.multinomial(weight_neg, distances.size(0)) # Each row of weight_neg must not sum to 0
            dist_neg = torch.gather(distances, 1, indexes.detach()) # detach to indicate requires_grad = False
            dist_neg = dist_neg[:,:n_samples].mean(1) # Averaging over n_samples negatives
        else:
            dist_neg = torch.cat([distances.diag(1), distances.diag(-distances.size(1)+1)])

        # Quadruplet {
        mix_idxs = list(range(distances.size(1))) # Creating ordered indexes
        mix_idxs = mix_idxs[1:] + mix_idxs[:1] # Displacing by 1, different anchors
        indexes = dist_neg.data.new(mix_idxs).long()
        dist_neg = dist_neg[indexes] # Getting displaced negative pairs
        # } Quadruplet;

        cost = dist_pos - dist_neg + self.alpha
        cost[cost < 0] = 0
        loss = cost.mean()
        return loss

    def dist(self, input_1, input_2):
        input_1 = nn.functional.normalize(input_1)
        input_2 = nn.functional.normalize(input_2)
        return 1 - torch.mm(input_1, input_2.t())
