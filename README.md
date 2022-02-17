# Training OOD Detectors in their Natural Habitats

This is the official repository of [Training OOD Detectors in their Natural Habitats](https://arxiv.org/abs/2202.03299) by Julian Katz-Samuels, Julia Nakhleh,
Rob Nowak, and Yixuan Li. This method trains OOD detectors effectively using auxiliary data that may be a mixture of both
inlier and outlier examples.

# Pretrained models

You can find the pretrained models in 

```
./CIFAR/snapshots/pretrained
```

# Datasets

Download the data in the folder

```
./data
```

# Run

To run the code, execute 

```
bash run.sh score in_distribution aux_distribution test_distribution 
```

For example, to run woods on cifar10 using dtd as the mixture distribution and the test_distribution, execute

```
bash run.sh woods cifar10 dtd dtd 
```

# Main Files

* ```CIFAR/train.py``` contains the main code used to train model(s) under our framework.
* ```CIFAR/make_datasets.py``` contains the code for reading datasets into PyTorch.
* ```CIFAR/plot_results.py``` contains code for loading and analyzing experimental results.
* ```CIFAR/test.py``` contains code for testing experimental results in OOD setting.




# Datasets

Here are links for the less common outlier datasets used in the paper: [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html), [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz),
and [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy).


