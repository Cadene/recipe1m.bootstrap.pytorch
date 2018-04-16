import os
import torch
from bootstrap.lib.options import Options

def accuracy(output, target, topk=(1,), ignore_index=None):
    """Computes the precision@k for the specified values of k"""

    if ignore_index is not None:
        target_mask = (target != ignore_index)
        target = target[target_mask]
        output_mask = target_mask.unsqueeze(1)
        output_mask = output_mask.expand_as(output)
        output = output[output_mask]
        output = output.view(-1, output_mask.size(1))

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)[0])
    return res

def save_activation(identifier, activation):
    retrieval_dir = os.path.join(Options()['model']['metric']['retrieval_dir'],
                                 os.path.basename(Options()['exp']['dir']))
    if not os.path.exists(retrieval_dir):
        os.makedirs(retrieval_dir)
    file_path = os.path.join(retrieval_dir, '{}.pth'.format(identifier))
    torch.save(activation, file_path)

def load_activation(identifier):
    retrieval_dir = os.path.join(Options()['model']['metric']['retrieval_dir'],
                                 os.path.basename(Options()['exp']['dir']))
    if not os.path.exists(retrieval_dir):
        os.makedirs(retrieval_dir)
    file_path = os.path.join(retrieval_dir, '{}.pth'.format(identifier))
    return torch.load(file_path)

def delete_activation(identifier):
    retrieval_dir = os.path.join(Options()['model']['metric']['retrieval_dir'],
                                 os.path.basename(Options()['exp']['dir']))
    file_path = os.path.join(retrieval_dir, '{}.pth'.format(identifier))
    if os.path.isfile(file_path):
        os.remove(file_path)
