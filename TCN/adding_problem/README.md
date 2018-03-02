## The Adding Problem

### Overview

In this task, each input consists of a length-T sequence of depth 2, with all values randomly
chosen randomly in [0, 1] in dimension 1. The second dimension consists of all zeros except for
two elements, which are marked by 1. The objective is to sum the two random values whose second 
dimensions are marked by 1. One can think of this as computing the dot product of two dimensions.

Simply predicting the sum to be 1 should give an MSE of about 0.1767. 

### Data Generation

See `data_generator` in `utils.py`.

### Note

Because a TCN's receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence length T. 