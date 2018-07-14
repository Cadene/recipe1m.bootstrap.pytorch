import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics

class Trijoint(Model):

    def __init__(self, engine=None, cuda_tf=transforms.ToCuda):
        super(Trijoint, self).__init__(engine, cuda_tf=cuda_tf)
        self.network = networks.Trijoint()
        self.criterions = {}
        self.metrics = {}
        if 'train' in engine.dataset:
            self.criterions['train'] = criterions.Trijoint(engine)
            self.metrics['train'] = metrics.Trijoint(engine, mode='train')
        if 'eval' in engine.dataset:
            self.metrics['eval'] = metrics.Trijoint(engine, mode='eval')