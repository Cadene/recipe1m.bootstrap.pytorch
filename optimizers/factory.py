from bootstrap.lib.options import Options
from .trijoint import Trijoint

def factory(model, engine=None):

    if Options()['optimizer']['name'] == 'trijoint_fixed_fine_tune':
        optimizer = Trijoint(Options()['optimizer'], model, engine)
    else:
        raise ValueError()

    return optimizer

