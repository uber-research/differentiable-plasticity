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

Copyright (c) 2018-2019 Uber Technologies, Inc.

All code except for `awd-lstm` is licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the LICENSE file in this repository for the specific language governing 
permissions and limitations under the License. 

The `awd-lstm-lm` code is forked from the [Salesforce Language Modeling toolkit](https://github.com/salesforce/awd-lstm-lm), which implements the Merity et al. 2018 model. See NOTICE.md.
