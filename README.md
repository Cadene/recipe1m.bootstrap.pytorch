# recipe1m.bootstrap.pytorch

We are a [Machine Learning research team](https://mlia.lip6.fr/members) from Sorbonne University. Our goal for this project was to create a cross-modal retrieval system trained on the biggest dataset of cooking recipes. This kind of systems is able to retrieve the corresponding recipe given an image (food selfie), and the corresponding image from the recipe.

It was also the occasion to compare several state-of-the-art metric learning loss functions in a new context. This first analysis gave us some idea on how to improve the generalization of our model. Following this, we wrote two research papers on a new model, called Adamine after Adaptive Mining, that add structure in the retrieval space:

- [Cross-Modal Retrieval in the Cooking Context: Learning Semantic Text-Image Embeddings (ACM SIGIR2018)](https://arxiv.org/abs/1804.11146)
- [Images & Recipes: Retrieval in the cooking context (IEEE ICDE2018, DECOR workshop)](https://arxiv.org/abs/1805.00900)


### Summary:

* [Introduction](#introduction)
    * [Recipe-to-Image retrieval task](#recipe-to-image-retrieval-task)
    * [Quick insight about AdaMine](#quick-insight-about-our-adamine)
* [Installation](#installation)
    * [Install python3](#install-python3)
    * [Clone bootstrap.pytorch](#clone-boostrap-pytorch)
    * [Download dataset](#download-dataset)
* [Quick start](#quick-start)
    * [Train a model](#train-a-model-on-the-train-val-sets)
    * [Evaluate a model](#evaluate-a-model-on-the-test-set)
* [Available (pretrained) models](#available-pretrained-models)
    * [PWC](#pwc)
    * [PWC++ (Ours)](#pwc-ours)
    * [VSE](#vse)
    * [VSE++](#vse)
    * [AdaMine_avg (Ours)](#adamine-avg-ours)
    * [Lifted structure](#lifted-structure)
* [Documentation](#documentation)
* [Useful commands](#useful-commands)
    * [Compare experiments](#compare-experiments)
    * [Use a specific GPU](#use-a-specific-gpu)
    * [Overwrite an option](#overwrite-an-option)
    * [Resume training](#resume-training)
    * [Evaluate with 10k setup](#evaluate-with-10k-setup)
    * [API](#api)
    * [Extract your own image features](#extract-your-own-image-features)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)


## Introduction

### Recipe-to-Image retrieval task

<p align="center">
    <img src="https://github.com/Cadene/recipe1m.bootstrap.pytorch/raw/master/images/task.png" width="800"/>
</p>

Given a list of ingredients and a sequence of cooking instructions, the goal is to train a statistical model to retrieve the associated image. For each recipe, the top row indicates the top 5 images retrieved by our AdaMine model, and the bottom row, by a strong baseline.

### Quick insight about AdaMine

<p align="center">
    <img src="https://github.com/Cadene/recipe1m.bootstrap.pytorch/raw/master/images/model.png" width="500"/>
</p>

Features embedding

- The list of ingredients is embedded using a bi-LSTM.
- The sequence of instructions is embedded using a hierarchical LSTM (a LSTM to embed sentences word-by-word, a second LSTM to embed the outputs of the first one).
- Both ingredients and instructions representations are concatenated and embedded once again.
- The image is embedded using a ResNet101.

Metric learning:

- The cross-modal (texts and images) retrieval space is learned through a joint retrieval and classification loss.
- Aligning items according to a retrieval task allows capturing the fine-grained semantics of items.
- Aligning items according to class meta-data (ex: hamburger, pizza, cocktail, ice-cream) allows capturing the high-level semantic information.
- Both retrieval and classification losses are based on a triplet loss (VSE), which is improved by our proposed Adaptive Mining (AdaMine) strategy for efficient negative sampling.

Negative sampling strategy

- The classic triplet loss strategy takes all negative samples into account to calculate the error. However, this tends to produce a vanishing gradient.
- The recent (VSE++) strategy only takes the hard negative sample. It is usually efficient, but does not allow the model to converge on this dataset.
- Our AdaMine strategy takes into account informative samples only (i.e., non-zero loss). It corresponds to a smooth curriculum learning, starting with the classic strategy and ending with the hard samples, but without the burden of switching between strategies. AdaMine also controls the trade-off between the retrieval and classification losses along the training.


## Installation

### 1. Install python 3

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

```
conda create --name recipe1m python=3.7
source activate recipe1m
```

### 2. Fork/clone bootstrap.pytorch and this repo

We use a [high level framework](https://github.com/Cadene/bootstrap.pytorch.git) to be able to focus on the model instead of boilerplate code.

```
cd $HOME
git clone https://github.com/Cadene/bootstrap.pytorch.git recipe1m.project
cd recipe1m.project
pip install -r requirements.txt
git clone https://github.com/Cadene/recipe1m.bootstrap.pytorch.git recipe1m
pip install -r recip1m/requirements.txt
```

### 3. Download dataset

Please, create an account on http://im2recipe.csail.mit.edu/ and agree to the terms of use. This dataset was made for research and not for commercial use.

```
# DATA=data/recip1m
mkdir $DATA
cd $DATA
wget http://data.lip6.fr/cadene/recipe1m/bigrams1M.pkl
wget http://data.lip6.fr/cadene/recipe1m/classes1M.pkl
wget http://data.lip6.fr/cadene/recipe1m/data_lmdb.tar
wget http://data.lip6.fr/cadene/recipe1m/food101_classes_renamed.txt
wget http://data.lip6.fr/cadene/recipe1m/recipe1M.tar.gz
wget http://data.lip6.fr/cadene/recipe1m/remove1M.txt
wget http://data.lip6.fr/cadene/recipe1m/text.tar.gz
wget http://data.lip6.fr/cadene/recipe1m/titles1M.txt
tar -xvf data_lmdb.tar
rm data_lmdb.tar
tar -xzvf recipe1M.tar.gz
rm recipe1M.tar.gz
tar -xzvf text.tar.gz
rm text.tar.gz
```

Note: Features extracted from resnet50 are included in data_lmdb.


## Quick start

### Train a model on the train/val sets

The [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) file load the options contained in a yaml file, create the corresponding experiment directory (in logs/recipe1m) and start the training procedure.

For instance, you can train our best model by running:
```
python -m bootstrap.run -o recipe1m/options/adamine.yaml
```
Then, several files are going to be created:
- options.yaml (copy of options)
- logs.txt (history of print)
- logs.json (batchs and epochs statistics)
- view.html (learning curves)
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_eval_epoch.metric.recall_at_1_im2recipe_mean_engine.pth.tar (checkpoints of best epoch)
- ckpt_best_eval_epoch.metric.recall_at_1_im2recipe_mean_model.pth.tar
- ckpt_best_eval_epoch.metric.recall_at_1_im2recipe_mean_optimizer.pth.tar

Many loss functions are available in the `recipe1m/options` directory.

### Evaluate a model on the test set

At the end of the training procedure, you can evaluate your model on the testing set. In this example, [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) load the options from your experiment directory, resume the best checkpoint on the validation set and start an evaluation on the testing set instead of the validation set while skipping the training set (train_split is empty).
```
python -m bootstrap.run \
-o logs/recipe1m/adamine/options.yaml \
--exp.resume best_eval_epoch.metric.recall_at_1_im2recipe_mean \
--dataset.train_split \
--dataset.eval_split test
```

Note: by default, the model is evaluated on the 1k setup; more info on the 10k setup [here]()


## Available (pretrained) models

### PWC

Pairwise loss [[paper]](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
```
python -m bootstrap.run -o recipe1m/options/pairwise.yaml
```

### PWC++ (Ours)

Pairwise with positive and negative margins loss
```
python -m bootstrap.run -o recipe1m/options/pairwise_plus.yaml
```

### VSE

Triplet loss (VSE) [[paper]](http://www.jmlr.org/papers/volume10/weinberger09a/weinberger09a.pdf)
```
python -m bootstrap.run -o recipe1m/options/avg_nosem.yaml
```

### VSE++

Triplet loss with hard negative mining [[paper]](https://arxiv.org/abs/1707.05612)
```
python -m bootstrap.run -o recipe1m/options/max.yaml
```

### AdaMine_avg (Ours)

Triplet loss with semantic loss
```
python -m bootstrap.run -o recipe1m/options/avg.yaml
```

### AdaMine (Ours)

Triplet loss with semantic loss and adaptive sampling
```
python -m bootstrap.run -o recipe1m/options/adamine.yaml
```

Features from testing set:

```
# LOGS=/local/cadene/logs/recip1m
cd $LOGS
wget http://data.lip6.fr/cadene/im2recipe/logs/adamine.tar.gz
tar -xzvf adamine.tar.gz
```

### Lifted structure
Lifted structure loss [[paper]](https://arxiv.org/abs/1511.06452)
```
python -m bootstrap.run -o recipe1m/options/lifted_struct.yaml
```



## Documentation

```
TODO
```



## Useful commands

### Compare experiments

```
python -m bootstrap.compare -d \
logs/recipe1m/adamine \
logs/recipe1m/avg \
-k eval_epoch.metric.recall_at_1_im2recipe_mean max
```

Results:
```
## eval_epoch.metric.recall_at_1_im2recipe_mean

  Place  Method      Score    Epoch
-------  --------  -------  -------
      1  adamine    0.3827       76
      2  avg        0.3201       51
```

### Use a specific GPU

```
CUDA_VISIBLE_DEVICES=0 python -m boostrap.run -o options/recipe1m/adamine.yaml
```

### Overwrite an option

The boostrap.pytorch framework makes it easy to overwrite a hyperparameter. In this example, I run an experiment with a non-default learning rate. Thus, I also overwrite the experiment directory path:
```
python -m bootstrap.run -o recipe1m/options/adamine.yaml \
--optimizer.lr 0.0003 \
--exp.dir logs/recipe1m/adamine_lr,0.0003
```

### Resume training

If a problem occurs, it is easy to resume the last epoch by specifying the options file from the experiment directory while overwritting the `exp.resume` option (default is None):
```
python -m bootstrap.run -o logs/recipe1m/adamine/options.yaml \
--exp.resume last
```

### Evaluate with the 10k setup

Just as with the [1k setup](#evaluate-a-model-on-the-test-set), we load the best checkpoint. This time we also overwrite some options. The metrics will be displayed on your terminal at the end of the evaluation.

```
python -m bootstrap.run \
-o logs/recipe1m/adamine/options.yaml \
--exp.resume best_eval_epoch.metric.recall_at_1_im2recipe_mean \
--dataset.train_split \
--dataset.eval_split test \
--model.metric.nb_bags 5 \
--model.metric.nb_matchs_per_bag 10000
```

Note: Metrics can be stored in a json file by adding the `--misc.logs_name eval,test10k` option. It will create a `logs_eval,test10k.json` in your experiment directory.

### API

```
TODO
```

### Extract your own image features

```
TODO
```


## Citation

```
@inproceddings{carvalho2018cross,
  title={Cross-Modal Retrieval in the Cooking Context: Learning Semantic Text-Image Embeddings},
  author={Carvalho, Micael and Cad{\`e}ne, R{\'e}mi and Picard, David and Soulier, Laure and Thome, Nicolas and Cord, Matthieu},
  booktitle={The ACM conference on Research and Development in Information Retrieval (SIGIR)},
  year={2018},
  url={https://arxiv.org/abs/1804.11146}
}
```


## Acknowledgment

Special thanks to the authors of [im2recipe](http://im2recipe.csail.mit.edu) who developped Recip1M, the dataset used in this research project.
