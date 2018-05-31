# Pattern memorization and completion

This code implements the pattern completion task. Five binary pattern of 1000 elements are shown once each, and then a degraded copy of one of these patterns (with half the elements zeroed out) is presented and must be completed.

The `simplest.py` program is a deliberately simple, but fully functional implementation of this task with a recurrent plastic network. This program is designed to provide an easily understood example for  differentiable plasticity. It requires PyTorch, but does not use a GPU.

`simple.py` is a slightly more elaborate version that can make use of a GPU.

The `full.py` and `lstm.py` programs have more options and can be used to compare different architectures.
