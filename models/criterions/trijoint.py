import torch
import torch.nn as nn
from torch.autograd import Variable
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.lib.utils import merge_dictionaries
from .triplet import Triplet

class Trijoint(nn.Module):

    def __init__(self, engine=None):
        super(Trijoint, self).__init__()
        self.with_classif = Options()['model']['with_classif']
        if self.with_classif:
            self.weight_classif = Options()['model']['criterion']['weight_classif']
            if self.weight_classif == 0:
                Logger()('You should use "--model.with_classif False"', Logger.ERROR)
            self.weight_retrieval = 1 - 2 * Options()['model']['criterion']['weight_classif']

        if 'keep_background' in Options()['model']['criterion'] and Options()['model']['criterion']['keep_background']:
            self.keep_background = True
            self.ignore_index = -100 # http://pytorch.org/docs/master/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss
        else:
            self.keep_background = False
            self.ignore_index = 0

        Logger()('ignore_index={}'.format(self.ignore_index))
        if self.with_classif:
            self.criterion_image_classif = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            self.criterion_recipe_classif = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.retrieval_strategy = Options()['model']['criterion']['retrieval_strategy']['name']
        #parameters = {'margin': Options()['model']['criterion']['retrieval_strategy']['margin']}

        if self.retrieval_strategy == 'triplet':
            self.criterion_retrieval = Triplet(engine)

        # elif loss_name == 'quadruplet':
        #     self.criterion_retrieval = Quadruplet()

        # elif loss_name == 'pairwise':
        #     self.criterion_retrieval = Pairwise()

        # elif loss_name == 'pairwise_pytorch':
        #     self.criterion_retrieval = nn.CosineEmbeddingLoss()

        else:
            raise ValueError('Unknown loss ({})'.format(self.retrieval_strategy))

    def forward(self, activations, batch):
        out = self.criterion_retrieval(activations['image_embedding'],
                                       activations['recipe_embedding'],
                                       batch['match'],
                                       batch['image']['class_id'],
                                       batch['recipe']['class_id'])
        if self.with_classif:
            out['image_classif'] = self.criterion_image_classif(activations['image_classif'],
                                                                batch['image']['class_id'].squeeze())
            out['recipe_classif'] = self.criterion_recipe_classif(activations['recipe_classif'],
                                                                  batch['recipe']['class_id'].squeeze())
            out['loss'] *= self.weight_retrieval
            out['loss'] += out['image_classif'] * self.weight_classif
            out['loss'] += out['recipe_classif'] * self.weight_classif

        # for key, value in out.items():
        #     Logger().log_value('{}.batch.criterion.{}'.format(self.mode,key), value, should_print=True)

        return out
