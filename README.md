## Differentiable plasticity

This repo contains implementations of the algorithms described in [Differentiable plasticity: training plastic networks with gradient descent](https://arxiv.org/abs/1804.02464), a research paper from Uber AI Labs.

NOTE: please see also our more recent work on differentiable *neuromodulated* plasticity: the "[backpropamine](https://github.com/uber-research/backpropamine)" framework.

There are four different experiments included here:

- `simple`: Binary pattern memorization and completion. Read this one first!
- `images`: Natural image memorization and completion
- `omniglot`: One-shot learning in the Omniglot task
- `maze`: Maze exploration task (reinforcement learning)


We strongly recommend studying the `simple/simplest.py` program first, as it is deliberately kept as simple as possible while showing full-fledged differentiable plasticity learning.

The code requires Python 3 and PyTorch 0.3.0 or later. The `images` code also requires scikit-learn. By default our code requires a GPU, but most programs can be run on CPU by simply uncommenting the relevant lines (for others, remove all occurrences of `.cuda()`).

To comment, please open an issue. We will not be accepting pull requests but encourage further study of this research. To learn more, check out our accompanying article on the [Uber Engineering Blog](https://eng.uber.com/differentiable-plasticity).


