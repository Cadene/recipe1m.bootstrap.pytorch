import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics

class Trijoint(Model):

    def __init__(self, engine=None, cuda_tf=transforms.ToCuda, variable_tf=transforms.ToVariable):
        super(Trijoint, self).__init__(engine, cuda_tf=cuda_tf, variable_tf=variable_tf)
        self.network = networks.Trijoint()
        self.criterions = {
            'train': criterions.Trijoint(engine)
        }
        self.metrics = {
            'train': metrics.Trijoint(engine, Options()['dataset']['train_split']),
            'eval': metrics.Trijoint(engine, Options()['dataset']['eval_split'])
        }