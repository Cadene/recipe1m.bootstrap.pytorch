# recipe1m.bootstrap.pytorch

The goal of this project is to create a cross-modal retrieval system for cooking recipes. This system is able to retrieve the corresponding recipe given an image (food selfie), and the corresponding image from the recipe. It was also the occasion to compare several state-of-the-art metric learning loss functions in a new context. This first analysis gave us some idea on how to improve the generalization of our model. Following this, we wrote two research papers on a new method, called Adamin after Adaptive Mining, that add structure in the retrieval space:

- [Cross-Modal Retrieval in the Cooking Context: Learning Semantic Text-Image Embeddings (ACM SIGIR2018)](https://arxiv.org/abs/1804.11146)
- [Images & Recipes: Retrieval in the cooking context (IEEE ICDE2018, DECOR workshop)](https://arxiv.org/abs/1805.00900)

If you would like to cite our paper, please do it as follow :)

```
@inproceddings{carvalho2018cross,
  title={Cross-Modal Retrieval in the Cooking Context: Learning Semantic Text-Image Embeddings},
  author={Carvalho, Micael and Cad{\`e}ne, R{\'e}mi and Picard, David and Soulier, Laure and Thome, Nicolas and Cord, Matthieu},
  booktitle={The ACM conference on Research and Development in Information Retrieval (SIGIR)},
  year={2018}
}
```

We thank the authors of [Im2recipe](http://im2recipe.csail.mit.edu) who developped Recip1M, the dataset used in this project.

## Introduction

### Recipe-to-Image retrieval task

<p align="center">
    <img src="https://raw.githubusercontent.com/Cadene/recip1m.bootstrap.pytorch/master/images/task.png" width="800"/>
</p>

Given a list of ingredients and a sequence of cooking instructions, the goal is to train a statistical model to retrieve the associated image. For each recipe, the top row indicates the top 5 images retrieved by our AdaMine model, and the bottom row, by a strong baseline.

### AdaMine model

<p align="center">
    <img src="https://raw.githubusercontent.com/Cadene/recip1m.bootstrap.pytorch/master/images/model.png" width="500"/>
</p>

Features embedding:

- The list of ingredients is embedded using a bi-LSTM.
- The sequence of instructions is embedded using a hierarchical LSTM (a LSTM to embed sentences word-by-word, a second LSTM to embed the outputs of the first one).
- Both ingredients and instructions representations are concatenated and embedded once again.
- The image is embedded using a ResNet101.

Metric learning:

- The cross-modal (texts and images) retrieval space is learned through a joint retrieval and classification loss.
- Aligning items according to a retrieval task allows
capturing the fine-grained semantics of items.
- Aligning items according to class meta-data (ex: hamburger, pizza, cocktail, ice-cream) allows
capturing the high-level semantic information.
- Both retrieval and classification losses are based on a triplet loss (VSE), which is improved by our proposed Adaptive Mining (AdaMine) strategy for efficient negative sampling.

Negative sampling strategy:

- The classic triplet loss strategy takes all negative samples into account to calculate the error. However, this tends to produce a vanishing gradient.
- The recent max (VSE++) strategy only takes the worst negative sample. It is usually efficient, but does not allow the model to converge on this dataset.
- Our AdaMine strategy takes into account informative samples only (i.e., non-zero loss).


## Install 


### 1. Install python 3

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

```
conda create --name im2recipe python=3
source activate im2recipe
```

### 2. Fork/clone this repo

```
cd $HOME
git clone https://github.com/Cadene/recipe1m.bootstrap.pytorch.git 
cd recipe1m.bootstrap.pytorch
pip install -r requirements.txt
```

Note: You will need [bootstrap.pytorch](https://github.com/Cadene/bootstrap.pytorch), a high level framework for deep learning project, to run the code. It is included in the requirements text file.

### 3. Download the dataset

```
# DATA=/local/cadene/data/recip1m
cd $DATA
```

http://im2recipe.csail.mit.edu/



## Reproducing the results

### Basic scripts

Extract features from images:

```
TODO
```

Launch training:

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Display training curves:

```
google-chrome logs/2018-05-18_avg/view.html
```

Resume training (only if a problem occurs):

```
python -m bootstrap.run \
--path_opts logs/2018-05-18_avg/options.yaml \
--exp.resume last
```

Launch testing (at the end of the training):

```
python -m bootstrap.run \
--path_opts logs/2018-05-18_avg/options.yaml \
--exp.resume best_eval_epoch.recall@1 \
--dataset.train_split \
--dataset.eval_split test
```

### Available models

Pairwise loss:

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Triplet (VSE) loss:

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Triplet loss with semantic loss (ours):

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Triplet loss with semantic loss and adaptive sampling (ours):

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Triplet Max (VSE++) loss:

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```

Lifted structure loss:

```
python -m bootstrap.run \
--path_opts recipe1m/options/avg.yaml \
--exp.dir logs/2018-05-18_avg
```


