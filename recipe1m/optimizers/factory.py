from bootstrap.lib.options import Options
from .trijoint import Trijoint

def factory(model, engine):

    if Options()['optimizer']['name'] == 'trijoint_fixed_fine_tune':
        optimizer = Trijoint(model, engine)
    else:
        raise ValueError()

    return optimizer

