from bootstrap.lib.options import Options
from .trijoint import Trijoint

def factory(engine=None):

    if Options()['model']['name'] == 'trijoint':
        model = Trijoint(engine)
    else:
        raise ValueError()

    return model

