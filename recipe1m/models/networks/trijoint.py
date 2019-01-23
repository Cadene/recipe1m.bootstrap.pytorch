import sys
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        self.dim_out = self.resnet.last_linear.in_features
        self.resnet.last_linear = None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ImageEmbedding(nn.Module):

    def __init__(self, opt):
        super(ImageEmbedding, self).__init__()
        self.dim_emb = opt['dim_emb']
        self.activations = opt.get('activations', None)
        # modules
        self.convnet = ResNet()
        self.fc = nn.Linear(self.convnet.dim_out, self.dim_emb)

    def forward(self, image):
        x = self.convnet(image['data'])
        x = self.fc(x)
        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x


class RecipeEmbedding(nn.Module):

    def __init__(self, opt):
        super(RecipeEmbedding, self).__init__()
        self.path_ingrs = opt['path_ingrs']
        self.dim_ingr_out = opt['dim_ingr_out'] # 2048
        self.dim_instr_in = opt['dim_instr_in']
        self.dim_instr_out = opt['dim_instr_out']
        self.with_ingrs = opt['with_ingrs']
        self.with_instrs = opt['with_instrs']
        self.dim_emb = opt['dim_emb']
        self.activations = opt.get('activations', None)
        # modules
        if self.with_ingrs:
            self._make_emb_ingrs()
            self.rnn_ingrs = nn.LSTM(self.dim_ingr_in, self.dim_ingr_out,
                                     bidirectional=True, batch_first=True)
        if self.with_instrs:
            self.rnn_instrs = nn.LSTM(self.dim_instr_in, self.dim_instr_out,
                                      bidirectional=False, batch_first=True)
        self.fusion = 'cat'
        self.dim_recipe = 0
        if self.with_ingrs:
            self.dim_recipe += 2*self.dim_ingr_out
        if self.with_instrs:
            self.dim_recipe += self.dim_instr_out
        if self.dim_recipe == 0:
            Logger()('Ingredients or/and instructions must be embedded "--model.network.with_{ingrs,instrs} True"', Logger.ERROR)
        
        self.fc = nn.Linear(self.dim_recipe, self.dim_emb)

    def forward_ingrs_instrs(self, ingrs_out=None, instrs_out=None):
        if self.with_ingrs and self.with_instrs:
            if self.fusion == 'cat':
                fusion_out = torch.cat([ingrs_out, instrs_out], 1)
            else:
                raise ValueError()
            x = self.fc(fusion_out)
        elif self.with_ingrs:
            x = self.fc(ingrs_out)
        elif self.with_instrs:
            x = self.fc(instrs_out)

        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x

    def forward(self, recipe):
        if self.with_ingrs:
            ingrs_out = self.forward_ingrs(recipe['ingrs'])
        else:
            ingrs_out = None

        if self.with_instrs:
            instrs_out = self.forward_instrs(recipe['instrs'])
        else:
            instrs_out = None

        x = self.forward_ingrs_instrs(ingrs_out, instrs_out)        
        return x

    def _make_emb_ingrs(self):
        with open(self.path_ingrs, 'rb') as fobj:
            data = pickle.load(fobj)
        
        self.nb_ingrs = data[0].size(0)
        self.dim_ingr_in = data[0].size(1)
        self.emb_ingrs = nn.Embedding(self.nb_ingrs, self.dim_ingr_in)

        state_dict = {}
        state_dict['weight'] = data[0]
        self.emb_ingrs.load_state_dict(state_dict)

        # idx+1 because 0 is padding
        # data[1] contains idx_to_name_ingrs (look in datasets.Recipes(HDF5))
        #self.idx_to_name_ingrs = {idx+1:name for idx, name in enumerate(data[1])}

    # def _process_lengths(self, tensor):
    #     max_length = tensor.data.size(1)
    #     lengths = list(max_length - tensor.data.eq(0).sum(1).sequeeze())
    #     return lengths

    def _sort_by_lengths(self, ingrs, lengths):
        sorted_ids = sorted(range(len(lengths)),
                            key=lambda k: lengths[k],
                            reverse=True)
        sorted_lengths = sorted(lengths, reverse=True)
        unsorted_ids = sorted(range(len(lengths)),
                              key=lambda k: sorted_ids[k])
        sorted_ids = torch.LongTensor(sorted_ids)
        unsorted_ids = torch.LongTensor(unsorted_ids)
        if ingrs.is_cuda:
            sorted_ids = sorted_ids.cuda()
            unsorted_ids = unsorted_ids.cuda()
        ingrs = ingrs[sorted_ids]
        return ingrs, sorted_lengths, unsorted_ids

    def forward_ingrs(self, ingrs):
        # TODO: to put in dataloader
        #lengths = self._process_lengths(ingrs)
        sorted_ingrs, sorted_lengths, unsorted_ids = self._sort_by_lengths(
            ingrs['data'], ingrs['lengths'])

        emb_out = self.emb_ingrs(sorted_ingrs)
        pack_out = nn.utils.rnn.pack_padded_sequence(emb_out,
            sorted_lengths, batch_first=True)

        rnn_out, (hn, cn) = self.rnn_ingrs(pack_out)
        batch_size = hn.size(1)
        hn = hn.transpose(0,1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, self.dim_ingr_out*2)
        #hn = torch.cat(hn, 2) # because bidirectional
        #hn = hn.squeeze(0)
        hn = hn[unsorted_ids]
        return hn

    def forward_instrs(self, instrs):
        # TODO: to put in dataloader
        sorted_instrs, sorted_lengths, unsorted_ids = self._sort_by_lengths(
            instrs['data'], instrs['lengths'])
        pack_out = nn.utils.rnn.pack_padded_sequence(sorted_instrs,
            sorted_lengths, batch_first=True)

        rnn_out, (hn, cn) = self.rnn_instrs(sorted_instrs)
        hn = hn.squeeze(0)
        hn = hn[unsorted_ids]
        return hn

    def forward_one_ingr(self, ingrs, emb_instrs=None):
        emb_ingr = self.forward_ingrs(ingrs)
        if emb_instrs is None:
            emb_instrs = torch.zeros(1,self.dim_instr_out)
        if emb_ingr.is_cuda:
            emb_instrs = emb_instrs.cuda()

        fusion_out = torch.cat([emb_ingr, emb_instrs], 1)
        x = self.fc(fusion_out)

        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)

        return x


class Trijoint(nn.Module):

    def __init__(self, opt, nb_classes, with_classif=False):
        super(Trijoint, self).__init__()
        self.dim_emb = opt['dim_emb']
        self.nb_classes = nb_classes
        self.with_classif = with_classif
        # modules
        self.image_embedding = ImageEmbedding(opt)
        self.recipe_embedding = RecipeEmbedding(opt)

        if self.with_classif:
            self.linear_classif = nn.Linear(self.dim_emb, self.nb_classes)

    def get_parameters_recipe(self):
        params = []
        params.append({'params': self.recipe_embedding.parameters()})
        if self.with_classif:
            params.append({'params': self.linear_classif.parameters()})
        params.append({'params': self.image_embedding.fc.parameters()})
        return params

    def get_parameters_image(self):
        return self.image_embedding.convnet.parameters()

    def forward(self, batch):
        out = {}
        out['image_embedding'] = self.image_embedding(batch['image'])
        out['recipe_embedding'] = self.recipe_embedding(batch['recipe'])

        if self.with_classif:
            out['image_classif'] = self.linear_classif(out['image_embedding'])
            out['recipe_classif'] = self.linear_classif(out['recipe_embedding'])

        return out
