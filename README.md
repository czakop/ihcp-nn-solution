# Neural network solution of the inverse heat conduction problem
This repository contains the code used in [my thesis](https://github.com/czakop/ihcp-nn-solution/blob/master/thesis.pdf).

## Abstract
This thesis presents a novel approach to address the Inverse Heat Conduction
Problem (IHCP) by designing a neural network architecture capable of solving the
problem on different scales. Various network architectures were explored using the
analogy of finding inclusions in cast metal based on temperature measurements. The
results showed that convolutional architectures outperformed Multilayer Perceptrons
in solving the IHCP, achieving accurate predictions on coarser validation datasets.
However, poor generalization to higher-resolution data was observed due to lim-
ited training data variability and the risk of overfitting. To overcome this challenge,
a multigrid-inspired approach was proposed, incorporating an autoencoder-based
network. The autoencoder learned a compressed representation of the input data,
symbolizing the coarser scale, and this representation was then used in a pretrained
convolutional network for improved performance on finer data. The proposed ar-
chitecture demonstrated enhanced performance on higher-resolution data, leverag-
ing the multigrid-inspired restriction operator provided by the autoencoder. This
research contributes to the field of deep learning by addressing the challenges of
transferring knowledge across scales and improving the generalization of neural net-
works.

## Research objectives
The main objective of this study is to develop a neural network architecture
that is able to detect inclusions in cast metal cubes based on surface temperature
measurements, that is solving the inverse heat conduction problem. Specifically, we
aim to:
1. Simulate directly the heat conduction in a heterogeneous metal cube which is
heated from below and has insulated sides, so that obtain the temperature on
its top plane ([simulation.py](https://github.com/czakop/ihcp-nn-solution/blob/master/src/simulation.py) and [data_generation.ipynb](https://github.com/czakop/ihcp-nn-solution/blob/master/src/data_generation.ipynb)).
2. Develop a neural network that can accurately detect the size and position of
the inclusions based on input temperature measurements on the top face ([hc_utils.py](https://github.com/czakop/ihcp-nn-solution/blob/master/src/hc_utils.py) and [base_network.ipynb](https://github.com/czakop/ihcp-nn-solution/blob/master/src/base_network.ipynb)).
4. Provide a solution for detecting inclusions based on data measured on a finer
grid, which is more computationally expensive to simulate directly. In this case,
only coarse data is available to train the model, thus making the generalization
even more difficult ([hc_utils.py](https://github.com/czakop/ihcp-nn-solution/blob/master/src/hc_utils.py) and [refinement.ipynb](https://github.com/czakop/ihcp-nn-solution/blob/master/src/refinement.ipynb)).

## Designed model architecture
### High level architecture
![high_level_architecture](https://github.com/czakop/ihcp-nn-solution/blob/master/img/high_level_architecture.png)

### Convolutional block
![conv_block](https://github.com/czakop/ihcp-nn-solution/blob/master/img/conv_block.png)

### Training architecture
![training_architecture](https://github.com/czakop/ihcp-nn-solution/blob/master/img/training_architecture.png)

### Autoencoder
![autoencoder](https://github.com/czakop/ihcp-nn-solution/blob/master/img/autoencoder.png)
