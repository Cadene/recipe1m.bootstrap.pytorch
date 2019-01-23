import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics

class Trijoint(Model):

    def __init__(self,
                 opt,
                 nb_classes,
                 modes=['train', 'eval'],
                 engine=None,
                 cuda_tf=transforms.ToCuda):
        super(Trijoint, self).__init__(engine, cuda_tf=cuda_tf)

        self.network = networks.Trijoint(
            opt['network'],
            nb_classes,
            with_classif=opt['with_classif'])

        self.criterions = {}
        self.metrics = {}

        if 'train' in modes:
            self.criterions['train'] = criterions.Trijoint(
                opt['criterion'],
                nb_classes,
                opt['network']['dim_emb'],
                with_classif=opt['with_classif'],
                engine=engine)

            self.metrics['train'] = metrics.Trijoint(
                opt['metric'],
                with_classif=opt['with_classif'],
                engine=engine,
                mode='train')

        if 'eval' in modes:
            self.metrics['eval'] = metrics.Trijoint(
                opt['metric'],
                with_classif=opt['with_classif'],
                engine=engine,
                mode='eval')