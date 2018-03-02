## Copying Memory Task

### Overview

In this task, each input sequence has length T+20. The first 10 values are chosen randomly 
among the digits 1-8, with the rest being all zeros, except for the last 11 entries that are 
filled with the digit ‘9’ (the first ‘9’ is a delimiter). The goal is to generate an output 
of same length that is zero everywhere, except the last 10 values after the delimiter, where 
the model is expected to repeat the 10 values it encountered at the start of the input.

### Data Generation

See `data_generator` in `utils.py`.

### Note

- Because a TCN's receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence length T+20. 

- Using the `--seq_len` flag, one can change the # of values to recall (the typical setup is 10).