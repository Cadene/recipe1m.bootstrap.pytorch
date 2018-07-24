"""
# How to use api.py

```
$ python -m recipe1m.api -o logs/recipe1m/adamine/options.yaml \
--dataset.train_split \
--dataset.eval_split test \
--exp.resume best_eval_epoch.metric.med_im2recipe_mean \
--dataset.eval_split test \
--misc.logs_name api
```
"""

import os
import re
import json
import base64
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import bootstrap.lib.utils as utils
import bootstrap.engines as engines
import bootstrap.models as models
import bootstrap.datasets as datasets
from PIL import Image
from io import BytesIO
from glob import glob
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import init_logs_options_files
from bootstrap.run import main
from .models.metrics.trijoint import fast_distance


@Request.application
def application(request):
    if 'image' in request.form:
        pil_img = decode_image(request.form['image'])
        out = process_image(pil_img)
        answer = json.dumps(out)
        response = Response(answer)
    else:
        response = Response('Image missing')
        
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('X-XSS-Protection', '0')    
    return response


def decode_image(strb64):
    strb64 = re.sub('^data:image/.+;base64,', '', strb64)
    pil_img = Image.open(BytesIO(base64.b64decode(strb64)))
    return pil_img


def encode_image(pil_img):
    buffer_ = BytesIO()
    pil_img.save(buffer_, format='PNG')
    img_str = base64.b64encode(buffer_.getvalue()).decode()
    img_str = 'data:image/png;base64,'+img_str
    return img_str


def process_image(pil_img):
    # TODO: save the image and the results

    tensor = engine.dataset['eval'].images_dataset.image_tf(pil_img)
    item = {'data': tensor}
    batch = engine.dataset['eval'].items_tf()([item])
    batch = engine.model.prepare_batch(batch)

    # TODO: add (for pytorch0.4)
    #with torch.no_grad():
    img_emb = engine.model.network.image_embedding(batch)

    distances = fast_distance(img_emb.data.cpu(), rcp_embs)
    values, ids = distances[0].sort()

    out = {}
    for i in range(5):
        idx = ids[i]
        item = engine.dataset['eval'][idx]
        info = {}
        info['class_name'] = item['recipe']['class_name']
        info['ingredients'] = item['recipe']['layer1']['ingredients']
        info['instructions'] = item['recipe']['layer1']['instructions']
        info['url'] = item['recipe']['layer1']['url']
        info['title'] = item['recipe']['layer1']['title']
        info['path_img'] = item['image']['path']
        info['img_strb64'] = encode_image(load_img(item['image']['path']))
        out['top{}'.format(i)] = info
    
    return out

def load_img(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def api(path_opts=None):
    global engine
    global rcp_embs
    global img_embs

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
    engine.model.eval()
    engine.resume()

    dir_extract = os.path.join(Options()['exp']['dir'], 'extract', Options()['dataset']['eval_split'])
    path_img_embs = os.path.join(dir_extract, 'image_emdeddings.pth')
    path_rcp_embs = os.path.join(dir_extract, 'recipe_emdeddings.pth')
    img_embs = torch.load(path_img_embs)
    rcp_embs = torch.load(path_rcp_embs)

    # BEGIN DEBUG
    # pil_img = load_img(engine.dataset['eval'][100]['image']['path'])
    # out = process_image(pil_img)
    # END DEBUG

    # TODO: add this as cli?
    my_local_ip = 'localhost' #'192.168.0.41'
    my_local_port = 8080 #3456
    run_simple(my_local_ip, my_local_port, application)


if __name__ == '__main__':
    main(run=api)