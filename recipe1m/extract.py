"""
# How to use extract.py

```
$ python -m recipe1m.extract -o logs/recipe1m/adamine/options.yaml \
--dataset.train_split \
--dataset.eval_split test \
--exp.resume best_eval_epoch.metric.med_im2recipe_mean \
--misc.logs_name extract_test
```
"""

import os
import torch
import torch.backends.cudnn as cudnn
import bootstrap.lib.utils as utils
import bootstrap.engines as engines
import bootstrap.models as models
import bootstrap.datasets as datasets
import bootstrap.views as views
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import init_logs_options_files

def extract(path_opts=None):
    Options(path_opts)
    utils.set_random_seed(Options()['misc']['seed'])

    assert Options()['dataset']['eval_split'] is not None, 'eval_split must be set'
    assert Options()['dataset']['train_split'] is None, 'train_split must be None'

    init_logs_options_files(Options()['exp']['dir'], Options()['exp']['resume'])

    Logger().log_dict('options', Options(), should_print=True)
    Logger()(os.uname())
    if torch.cuda.is_available():
        cudnn.benchmark = True
        Logger()('Available GPUs: {}'.format(utils.available_gpu_ids()))

    engine = engines.factory()
    engine.dataset = datasets.factory(engine)
    engine.model = models.factory(engine)
    engine.view = views.factory(engine)

    # init extract directory
    dir_extract = os.path.join(Options()['exp']['dir'], 'extract', Options()['dataset']['eval_split'])
    os.system('mkdir -p '+dir_extract)
    path_img_embs = os.path.join(dir_extract, 'image_emdeddings.pth')
    path_rcp_embs = os.path.join(dir_extract, 'recipe_emdeddings.pth')
    img_embs = torch.FloatTensor(len(engine.dataset['eval']), Options()['model.network.dim_emb'])
    rcp_embs = torch.FloatTensor(len(engine.dataset['eval']), Options()['model.network.dim_emb'])

    def save_embeddings(module, input, out):
        nonlocal img_embs
        nonlocal rcp_embs
        batch = input[0] # tuple of len=1
        for j, idx in enumerate(batch['recipe']['index']):
            # path_image = os.path.join(dir_image, '{}.pth'.format(idx))
            # path_recipe = os.path.join(dir_recipe, '{}.pth'.format(idx))
            # torch.save(out['image_embedding'][j].data.cpu(), path_image)
            # torch.save(out['recipe_embedding'][j].data.cpu(), path_recipe)
            img_embs[idx] = out['image_embedding'][j].data.cpu()
            rcp_embs[idx] = out['recipe_embedding'][j].data.cpu()

    engine.model.register_forward_hook(save_embeddings)
    engine.resume()
    engine.eval()

    Logger()('Saving image embeddings to {} ...'.format(path_img_embs))
    torch.save(img_embs, path_img_embs)

    Logger()('Saving recipe embeddings to {} ...'.format(path_rcp_embs))
    torch.save(rcp_embs, path_rcp_embs)


if __name__ == '__main__':
    from bootstrap.run import main
    main(run=extract)