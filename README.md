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


=======
# Backpropamine: differentiable neuromodulated plasticity.

This code shows how to train neural networks with neuromodulated plastic connections, as described in [Backpropamine: training self-modifying
neural networks with differentiable neuromodulated plasticity (Miconi et al.
ICLR 2019)](https://openreview.net/pdf?id=r1lrAiA5Ym), a research paper from
Uber AI Labs.

## Tasks


This repository contains four different experiments:

* `simplemaze`: a simple implementation of the "grid maze" exploration task. If you want to understand how backpropamine works, read this one first!
* `maze`: the original implementation of the "grid maze" task, with more options.
* `sr`: a simple associative learning task in which the agent must learn to identify a certain unpredictable "target" stimulus, based solely on reward feedback.
* `awd-lstm`: an extension of the powerful language modeling system by [Merity et al. 2018](https://github.com/salesforce/awd-lstm-lm), using neuromodulated plastic LSTMs.

We strongly recommend studying the `simplemaze` experiment first, as it is deliberately kept as simple as possible while showing full-fledged backpropamine learning.

## Copyright and licensing information

Copyright (c) 2017-2019 Uber Technologies, Inc.

All code is licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the LICENSE file in this repository for the specific language governing 
permissions and limitations under the License. 

