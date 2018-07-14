from bootstrap.lib.options import Options
from .trijoint import Trijoint

def factory(engine=None):

    if Options()['model.name'] == 'trijoint':
        model = Trijoint(
            Options()['model'],
            Options()['dataset.nb_classes'],
            engine.dataset.keys(),
            engine)
    else:
        raise ValueError()

    return model

