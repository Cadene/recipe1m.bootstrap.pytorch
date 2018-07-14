import torch
import torch.nn as nn
from torch.autograd import Variable
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class Pairwise(nn.Module):

    def __init__(self, opt):
        super(Pairwise, self).__init__()
        self.alpha_pos = opt['retrieval_strategy']['pos_margin']
        self.alpha_neg = opt['retrieval_strategy']['neg_margin']

    def forward(self, input1, input2, target):
        # target should be 1 for matched samples or -1 for not matched ones
        distances = self.dist(input1, input2)
        target = target.squeeze(1)
        cost = target * distances

        cost[target > 0] -= self.alpha_pos
        cost[target < 0] += self.alpha_neg
        cost[cost < 0] = 0

        out = {}
        out['bad_pairs'] = (cost == 0).float().sum() / cost.numel()
        out['loss'] = cost.mean()
        return out

    def dist(self, input1, input2):
        input1 = nn.functional.normalize(input1)
        input2 = nn.functional.normalize(input2)
        return 1 - torch.mul(input1, input2).sum(1)
