from bootstrap.lib.options import Options
from .extract import Extract

def factory():

    if Options()['engine']['name'] == 'extract':
        return Extract()
    else:
        raise ValueError()