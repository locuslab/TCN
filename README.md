# Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN)


This repository contains the experiments done in the work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

We specifically target a comprehensive set of tasks that have been repeatedly used to compare the effectiveness of different recurrent networks, and evaluate a simple, generic but powerful (purely) convolutional network on the recurrent nets' home turf.

Experiments are done in PyTorch. If you find this repository helpful, please cite our work:

```
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
```

## Domains and Datasets

**Update**: The code should be directly runnable with PyTorch 0.4.0 or above (PyTorch 1.0 strongly recommended). The older versions of PyTorch are no longer supported.

This repository contains the benchmarks to the following tasks, with details explained in each sub-directory:

  - **The Adding Problem** with various T (we evaluated on T=200, 400, 600)
  - **Copying Memory Task** with various T (we evaluated on T=500, 1000, 2000)
  - **Sequential MNIST** digit classification
  - **Permuted Sequential MNIST** (based on Seq. MNIST, but more challenging)
  - **JSB Chorales** polyphonic music
  - **Nottingham** polyphonic music
  - **PennTreebank** [SMALL] word-level language modeling (LM)
  - **Wikitext-103** [LARGE] word-level LM
  - **LAMBADA** [LARGE] word-level LM and textual understanding
  - **PennTreebank** [MEDIUM] char-level LM
  - **text8** [LARGE] char-level LM

While some of the large datasets are not included in this repo, we use the [observations](https://github.com/edwardlib/observations) package to download them, which can be easily installed using pip. 

## Usage

Each task is contained in its own directory, with the following structure:

```
[TASK_NAME] /
    data/
    [TASK_NAME]_test.py
    models.py
    utils.py
```

To run TCN model on the task, one only need to run `[TASK_NAME]_test.py` (e.g. `add_test.py`). To tune the hyperparameters, one can specify via argument options, which can been seen via the `-h` flag. 
