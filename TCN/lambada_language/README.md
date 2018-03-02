## Word-level Language Modeling

### Overview

LAMBADA is a collection of narrative passages sharing the characteristics such that human subjects are able to guess accurately given sufficient context, but not so if they only see the last sentence containing the target word. On average, the context contains 4.6 sentences, and the testing performance is evaluated by having the model the last element of the target sentence (i.e. the very last word). 

Most of the existing computational models fail on this task (without the help of external memory unit, such as neural cache). See [the original LAMBADA paper](https://arxiv.org/pdf/1606.06031.pdf) for more results on applying RNNs on LAMBADA.

**Example**: 
```
Context: “Yes, I thought I was going to lose the baby.” “I was scared too,” he stated, sincerity flooding his eyes. “You were ?” “Yes, of course. Why do you even ask?” “This baby wasn’t exactly planned for.”

Target sentence: “Do you honestly think that I would want you to have a _______” 

Target word: miscarriage
```

### Data

See `data_generator` in `utils.py`. You will need to download the lambada dataset from [here](http://clic.cimec.unitn.it/lambada/) and put it under director `./data/lambada` (or other paths specified by `--data` flag). 


### Note

- Just like in a recurrent network implementation where it is common to repackage 
hidden units when a new sequence begins, we pass into TCN a sequence `T` consisting 
of two parts: 1) effective history `L1`, and 2) valid sequence `L2`:

```
Sequence [---------T--------->] = [--L1--> ------L2------>]
```

In the forward pass, the whole sequence is passed into TCN, but only the `L2` portion is used for 
training. This ensures that the training data are also provided with sufficient history. The size
of `T` and `L2` can be adjusted via flags `seq_len` and `validseqlen`. 

- The choice of data to load can be specified via the `--data` flag, followed by the path to
the directory containing the data. For instance, running

```
python lambada_test.py --data .data/lambada
```

would train on the LAMBADA (PTB) dataset, if it is contained in `.data/lambada`.

- LAMBADA is a huge dataset with lots of vocabularies. A