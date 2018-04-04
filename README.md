## Differentiable plasticity

This is the code for the upcoming paper "Differentiable plasticity: training plastic networks with gradient descent".

There are four different experiments included here:

- `simple`: Binary pattern memorization and completion. Read this one first!
- `images`: Natural image memorization and completion
- `omniglot`: One-shot learning in the Omniglot task
- `maze`: Maze exploration task (reinforcement learning)


We strongly recommend studying the `simple/simple.py` program first, as it is deliberately kept as simple as possible while showing full-fledged differentiable plasticity learning.

The code requires Python 3 and PyTorch 0.3.0 or later. The `images` code also requires scikit-learn. By default our code requires a GPU, but most programs can be run on CPU by simply uncommenting the relevant lines (for others, remove all occurrences of `.cuda()`).

This code is provided for informative purposes only.

