## Word-level Language Modeling

### Overview

In word-level language modeling tasks, each element of the sequence is a word, where the model
is expected to predict the next incoming word in the text. We evaluate the temporal convolutional
network as a word-level language model on three datasets: PennTreebank (PTB), Wikitext-103 
and LAMBADA.

Because the evaluation of LAMBADA has different requirement (predicting only the very last word
based on a broader context), we put it in another directory. See `../lambada_language`. 


### Data

- **PennTreebank**: A frequently studied, but still relatively
small language corpus. When used as a word-level language corpus,
PTB contains 888K words for training, 70K for validation,
and 79K for testing, with a vocabulary size of 10K.

- **Wikitext-103**: Wikitext-103 is almost
110 times as large as PTB, featuring a vocabulary size of
about 268K. The dataset contains 28K Wikipedia articles
(about 103 million words) for training, 60 articles (about
218K words) for validation, and 60 articles (246K words)
for testing. This is a more representative and realistic dataset
than PTB, with a much larger vocabulary that includes many
rare words, and have been used in (e.g. Merity et al. (2016)).

- **LAMBADA**: An even larger language corpus than Wikitext-103
consisting of novels from different categories. The goal is to 
test a model's ability to understand text and predict according
to a long context. See `../lambada_language`. 

See `data_generator` in `utils.py`.


### Note

- Just like in a recurrent network implementation where it is common to repackage 
hidden units when a new sequence begins, we pass into TCN a sequence `T` consisting 
of two parts: 1) effective history `L1`, and 2) valid sequence `L2`:

```
Sequence [---------T--------->] = [--L1--> ------L2------>]
```

In the forward pass, the whole sequence is passed into TCN, but only the `L2` portion is used for 
training. This ensures that the training data are also provided with sufficient history. The size
of `T` and `L2` can be adjusted via flags `seq_len` and `validseqlen`. A similar setting
was used in character-level language modeling experiments.

- The choice of data to load can be specified via the `--data` flag, followed by the path to
the directory containing the data. For instance, running

```
python word_cnn_test.py --data .data/penn
```

would train on the PennTreebank (PTB) dataset, if it is contained in `.data/penn`.