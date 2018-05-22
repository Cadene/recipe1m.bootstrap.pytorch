from bootstrap.lib.options import Options
from .recipe1m import Recipe1M

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'recipe1m':    
        
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_recipe1m(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None): 
            dataset['eval'] = factory_recipe1m(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_recipe1m(split):
    dataset = Recipe1M(
        Options()['dataset']['dir'],
        split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        freq_mismatch=Options()['dataset']['freq_mismatch'],
        batch_sampler=Options()['dataset']['batch_sampler'],
        image_from=Options()['dataset']['image_from'])
    return dataset