## Sequential MNIST & Permuted Sequential MNIST

### Overview

MNIST is a handwritten digit classification dataset (Lecun et al., 1998) that is frequently used to 
test deep learning models. In particular, sequential MNIST is frequently used to test a recurrent 
network’s ability to retain information from the distant past (see paper for references). In 
this task, each MNIST image (28 x 28) is presented to the model as a 784 × 1 sequence 
for digit classification. In the more challenging permuted MNIST (P-MNIST) setting, the order of 
the sequence is permuted at a (fixed) random order.

### Data

See `data_generator` in `utils.py`. You only need to download the data once. The default path
to store the data is at `./data/mnist`.

Original source of the data can be found [here](http://yann.lecun.com/exdb/mnist/).

### Note

- Because a TCN's receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence length 784. 

- While this is a sequence model task, we only use the very last output (i.e. at time T=784) for 
the eventual classification.