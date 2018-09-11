# recipe1m.bootstrap.pytorch

We are a [Machine Learning research team](https://mlia.lip6.fr/members) from Sorbonne University. Our goal for this project was to create a cross-modal retrieval system trained on the biggest dataset of cooking recipes. This kind of systems is able to retrieve the corresponding recipe given an image (food selfie), and the corresponding image from the recipe. It was also the occasion to compare several state-of-the-art metric learning loss functions in a new context. This first analysis gave us some idea on how to improve the generalization of our model. Following this, we wrote two research papers on a new model, called Adamin after Adaptive Mining, that add structure in the retrieval space:

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

Special thanks to the authors of [Im2recipe](http://im2recipe.csail.mit.edu) who developped Recip1M, the dataset used in this research project.

## Introduction

### Recipe-to-Image retrieval task

<p align="center">
    <img src="https://github.com/Cadene/recipe1m.bootstrap.pytorch/raw/master/images/task.png" width="800"/>
</p>

Given a list of ingredients and a sequence of cooking instructions, the goal is to train a statistical model to retrieve the associated image. For each recipe, the top row indicates the top 5 images retrieved by our AdaMine model, and the bottom row, by a strong baseline.

### AdaMine model

<p align="center">
    <img src="https://github.com/Cadene/recipe1m.bootstrap.pytorch/raw/master/images/model.png" width="500"/>
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
- The recent (VSE++) strategy only takes the hard negative sample. It is usually efficient, but does not allow the model to converge on this dataset.
- Our AdaMine strategy takes into account informative samples only (i.e., non-zero loss). It corresponds to a smooth curriculum learning, starting with the classic strategy and ending with the hard samples, but without the burden of switching between strategies. AdaMine also controls the trade-off between the retrieval and classification losses along the training.


## Install 


Please follow the instructions in: https://github.com/Cadene/recipe1m.bootstrap.pytorch/tree/pytorch0.2

/!\ We have a strange bug with versions of pytorch >0.2 which makes it impossible to converge. Please consider using pytorch0.2 instead.


## Acknowledgment

Special thanks to our professors and friends from our lab for the perfect working atmosphere.