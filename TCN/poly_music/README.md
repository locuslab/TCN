## Polyphonic Music Dataset

### Overview

We evaluate temporal convolutional network (TCN) on two popular polyphonic music dataset, described below.

- **JSB Chorales** dataset (Allan & Williams, 2005) is a polyphonic music dataset con-
sisting of the entire corpus of 382 four-part harmonized chorales by J. S. Bach. In a polyphonic
music dataset, each input is a sequence of elements having 88 dimensions, representing the 88 keys
on a piano. Therefore, each element `x_t` is a chord written in as binary vector, in which a “1” indicates
a key pressed.

- **Nottingham** dataset is a collection of 1200 British and American folk tunes. Not-
tingham is a much larger dataset than JSB Chorales. Along with JSB Chorales, Nottingham has
been used in a number of works that investigated recurrent models’ applicability in polyphonic mu-
sic, and the performance for both tasks are measured in terms
of negative log-likelihood (NLL) loss.

The goal here is to predict the next note given some history of the notes played.

### Data

See `data_generator` in `utils.py`. The data has been pre-processed and can be loaded directly using 
scipy functions.

Original source of the data can be found [here](http://www-etud.iro.umontreal.ca/~boulanni/icml2012).

### Note

- Each sequence can have a different length. In the current implementation, we simply train each
sequence separately (i.e. batch size is 1), but one can zero-pad all sequences to the same length
and train by batch.

- One can use different datasets by specifying through the `--data` flag on the command line. The
default is `Nott`, for Nottingham.

- While each data is binary, the fact that there are 88 dimensions (for 88 keys) means there are
essentially `2^88` "classes". Therefore, instead of directly predicting each key directly, we
follow the standard practice so that a sigmoid is added at the end of the network. This ensures
that every entry is converted to a value between 0 and 1 to compute the NLL loss.


