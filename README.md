# Backpropamine: differentiable neuromdulated plasticity.


This code shows how to train neural networks with neuromodulated plastic connections, as described in Miconi et al. ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym ).

## Task

For now, the published code only implements the "Grid Maze" task. See Section 4.5 in Miconi et al. 
ICML 2018 ( https://arxiv.org/abs/1804.02464 ), or Section 4.2 in
Miconi et al. ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym )

During each episode, the agent must explore a regular grid-shaped maze to find
a reward location. When the agent hits the reward, it is randomly teleported
within the maze. The agent's goal is to hit the reward as often as possible
within an episode (200 time steps by default). The reward location is randomly
chosen for each episode, but fixed within an episode. This, performing the task
requires finding the reward, memorizing its location and repeatedly navigating
back to it.

## Backpropamine network

The `Network` class in `maze/maze.py` implements a Backpropamine network, that is, a neural
network with neuromodulated Hebbian plastic connections that is trained by
gradient descent.  

Here is the full code for the `Network` class, which contains the entire machinery for Backpropamine (note that it only contains ~25 lines of code!)

```python
class Network(nn.Module):
    
    def __init__(self, isize, hsize): 
        super(Network, self).__init__()
        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline ("fixed") component of the plastic recurrent layer
        
        self.alpha =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection
        #self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))  # Per-neuron alpha
        #self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))         # Single alpha for whole network

        self.h2mod = torch.nn.Linear(hsize, 1)      # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(1, hsize)  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)

        self.h2o = torch.nn.Linear(hsize, NBACTIONS)    # From recurrent to outputs (action probabilities)
        self.h2v = torch.nn.Linear(hsize, 1)            # From recurrent to value-prediction (used for A2C)


        
    def forward(self, inputs, hidden): # hidden is a tuple containing h-state, the hebbian trace and the eligibility trace
            HS = self.hsize
        
            # hidden[0] is the h-state; hidden[1] is the Hebbian trace
            hebb = hidden[1]


            # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
            hactiv = torch.tanh( self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  )
            activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv.unsqueeze(1))  # Batched outer product of previous hidden state with new hidden state
            
            # We also need to compute the eta (the plasticity rate), wich is determined by neuromodulation
            myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
            
            # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
            myeta = self.modfanout(myeta) 
            
            
            # Updating Hebbian traces, with a hard clip (other choices are possible)
            self.clipval = 2.0
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

            hidden = (hactiv, hebb)
            return activout, valueout, hidden



    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)

    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

```


The rest of the code implements a simple
A2C algorithm to train the network for the Grid Maze task.

## Copyright and licensing information

Copyright (c) 2018-2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License file in this repository for the specific language governing 
permissions and limitations under the License.
