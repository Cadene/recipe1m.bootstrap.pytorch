from bootstrap.lib.options import Options
from .recipe1m import Recipe1M

def factory(split):
    if Options()['dataset']['name'] == 'recipe1m':
        
        dir_data = Options()['dataset']['dir']
        batch_size = Options()['dataset']['batch_size']
        nb_threads = Options()['dataset']['nb_threads']
        freq_mismatch = Options()['dataset']['freq_mismatch']
        batch_sampler = Options()['dataset']['batch_sampler']
        image_from = Options()['dataset']['image_from']

        dataset = Recipe1M(dir_data, split,
            batch_size=batch_size, nb_threads=nb_threads,
            freq_mismatch=freq_mismatch, batch_sampler=batch_sampler,
            image_from=image_from)
    else:
        raise ValueError()

    return dataset

