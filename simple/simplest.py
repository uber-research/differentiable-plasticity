# Differentiable plasticity: simplest fully functional code.

# Copyright (c) 2018 Uber Technologies, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# This is a very simple, but fully functional implementation of Differentiable
# Plasticity. It implements the binary pattern completion task discussed in
# Section 4.1 of Miconi et al. ICML 2018 (https://arxiv.org/abs/1804.02464).

# The code implements a simple RNN with plastic weights. It requires PyTorch,
# but does not use a GPU.

# The actual code that specifically implements plasticity
# amounts to less than 4 lines of code in total (see Section
# S1 in the paper cited above).


import argparse
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import time

PATTERNSIZE = 1000      # Size of the patterns to memorize 
NBNEUR = PATTERNSIZE    # One neuron per pattern element
NBPATTERNS = 5          # The number of patterns to learn in each episode
NBPRESCYCLES = 2        # Number of times each pattern is to be presented
PRESTIME = 6            # Number of time steps for each presentation
PRESTIMETEST = 6        # Same thing but for the final test pattern
INTERPRESDELAY = 4      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode


# Generate the full list of inputs, as well as the target output at last time step, for an episode. 
def generateInputsAndTarget():
    inputT = np.zeros((NBSTEPS, 1, NBNEUR)) #inputTensor, initially in numpy format
    # Create the random patterns to be memorized in an episode
    patterns=[]
    for nump in range(NBPATTERNS):
        patterns.append(2*np.random.randint(2, size=PATTERNSIZE)-1)
    # Building the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    degradedtestpattern = testpattern * np.random.randint(2, size=PATTERNSIZE)
    # Inserting the inputs in the input tensor at the proper places
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi =nc * (NBPATTERNS * (PRESTIME+INTERPRESDELAY)) + ii * (PRESTIME+INTERPRESDELAY) + nn
                inputT[numi][0][:] = patterns[ii][:]
    # Inserting the degraded pattern
    for nn in range(PRESTIMETEST):
        inputT[-PRESTIMETEST + nn][0][:] = degradedtestpattern[:]
    inputT = 20.0 * torch.from_numpy(inputT.astype(np.float32))  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern.astype(np.float32))
    return inputT, target

total_loss = 0.0; all_losses = []
nowtime = time.time()


# === Actual algorithm ===
# Note that each column of w and alpha defines the inputs to a single neuron
w = Variable(.01 * torch.randn(NBNEUR, NBNEUR), requires_grad=True) # Fixed weights
alpha = Variable(.01 * torch.randn(NBNEUR, NBNEUR), requires_grad=True) # Plasticity coeffs.
optimizer = torch.optim.Adam([w, alpha], lr=3e-4)

print("Starting episodes...")
for numiter in range(1000): # Loop over episodes
    y = Variable(torch.zeros(1, NBNEUR)) # Initialize neuron activations
    hebb = Variable(torch.zeros(NBNEUR, NBNEUR)) # Initialize Hebbian trace
    inputs, target = generateInputsAndTarget() # Generate inputs & target for this episode
    optimizer.zero_grad()
    # Run the episode:
    for numstep in range(NBSTEPS):
        yout = F.tanh( y.mm(w + torch.mul(alpha, hebb)) +
                Variable(inputs[numstep], requires_grad=False) )
        hebb = .99 * hebb + .01 * torch.ger(y[0], yout[0]) # torch.ger = Outer product
        y = yout
    # Episode done, now compute loss, apply backpropagation
    loss = (y[0] - Variable(target, requires_grad=False)).pow(2).sum()
    loss.backward()
    optimizer.step()

# === End of actual algorithm ===


    # Print statistics
    print_every = 10
    to = target.cpu().numpy(); yo = y.data.cpu().numpy()[0][:]
    z = (np.sign(yo) != np.sign(to)); lossnum = np.mean(z)  # Compute error rate
    total_loss  += lossnum
    if (numiter+1) % print_every == 0:
        previoustime = nowtime;  nowtime = time.time()
        print("Episode", numiter, "=== Time spent on last", print_every, "iters: ", nowtime - previoustime)
        print(target.cpu().numpy()[-10:])   # Target pattern to be reconstructed
        print(inputs.cpu().numpy()[numstep][0][-10:])  # Last input (degraded pattern)
        print(y.data.cpu().numpy()[0][-10:])   # Final output of the network
        total_loss /= print_every
        print("Mean error rate over last", print_every, "iters:", total_loss, "\n")
        total_loss = 0


