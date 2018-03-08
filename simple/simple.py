# Differentiable plasticity: simple binary pattern memorization and reconstruction.
#
# This program is meant as a simple example for differentiable plasticity. It is fully functional but not very flexible.
#
# Usage: python simple.py [rngseed], where rngseed is an optional parameter specifying the seed of the random number generator. 
# To use it on a GPU or CPU, toggle comments on the 'ttype' declaration below.

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle as pickle
import pdb
import time


PATTERNSIZE = 1000
NBNEUR = PATTERNSIZE+1  # NbNeur = Pattern Size + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
#ETA = .01               # The "learning rate" of plastic connections - we actually learn it
ADAMLEARNINGRATE =3e-4  # The learning rate of the Adam optimizer 
RNGSEED = 0             # Initial random seed - can be modified by passing a number as command-line argument

# Note that these patterns are likely not optimal
PROBADEGRADE = .5       # Proportion of bits to zero out in the target pattern at test time
NBPATTERNS = 5          # The number of patterns to learn in each episode
NBPRESCYCLES = 2        # Number of times each pattern is to be presented
PRESTIME = 6            # Number of time steps for each presentation
PRESTIMETEST = 6        # Same thing but for the final test pattern
INTERPRESDELAY = 4      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode


if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))
np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)


#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU


# Generate the full list of inputs for an episode. The inputs are returned as a PyTorch tensor of shape NbSteps x 1 x NbNeur
def generateInputsAndTarget():
    inputT = np.zeros((NBSTEPS, 1, NBNEUR)) #inputTensor, initially in numpy format...

    # Create the random patterns to be memorized in an episode
    seedp = np.ones(PATTERNSIZE); seedp[:PATTERNSIZE//2] = -1
    patterns=[]
    for nump in range(NBPATTERNS):
        p = np.random.permutation(seedp)
        patterns.append(p)

    # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
    # Choosing the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = random.choice(patterns).copy()
    preservedbits = np.ones(PATTERNSIZE); preservedbits[:int(PROBADEGRADE * PATTERNSIZE)] = 0; np.random.shuffle(preservedbits)
    degradedtestpattern = testpattern * preservedbits

    # Inserting the inputs in the input tensor at the proper places
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi =nc * (NBPATTERNS * (PRESTIME+INTERPRESDELAY)) + ii * (PRESTIME+INTERPRESDELAY) + nn
                inputT[numi][0][:PATTERNSIZE] = patterns[ii][:]

    # Inserting the degraded pattern
    for nn in range(PRESTIMETEST):
        inputT[-PRESTIMETEST + nn][0][:PATTERNSIZE] = degradedtestpattern[:]

    for nn in range(NBSTEPS):
        inputT[nn][0][-1] = 1.0  # Bias neuron.
        inputT[nn] *= 20.0       # Strengthen inputs
    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern).type(ttype)

    return inputT, target



class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron
        self.w = Variable(.01 * torch.randn(NBNEUR, NBNEUR).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights
        self.alpha = Variable(.01 * torch.randn(NBNEUR, NBNEUR).type(ttype), requires_grad=True)  # The matrix of plasticity coefficients
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

    def forward(self, input, yin, hebb):
        # Run the network for one timestep
        yout = F.tanh( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)
        return yout, hebb

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, NBNEUR).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return Variable(torch.zeros(NBNEUR, NBNEUR).type(ttype))


net = NETWORK()
optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=ADAMLEARNINGRATE)
total_loss = 0.0; all_losses = []
print_every = 10
nowtime = time.time()

for numiter in range(2000):
    # Initialize network for each episode
    y = net.initialZeroState()
    hebb = net.initialZeroHebb()
    optimizer.zero_grad()

    # Generate the inputs and target pattern for this episode
    inputs, target = generateInputsAndTarget()

    # Run the episode!
    for numstep in range(NBSTEPS):
        y, hebb = net(Variable(inputs[numstep], requires_grad=False), y, hebb)

    # Compute loss for this episode (last step only)
    loss = (y[0][:PATTERNSIZE] - Variable(target, requires_grad=False)).pow(2).sum()

    # Apply backpropagation to adapt basic weights and plasticity coefficients
    loss.backward()
    optimizer.step()


    # That's it for the actual algorithm!
    # Print statistics, save files
    #lossnum = loss.data[0]   # Saved loss is the actual learning loss (MSE)
    to = target.cpu().numpy(); yo = y.data.cpu().numpy()[0][:PATTERNSIZE]; z = (np.sign(yo) != np.sign(to)); lossnum = np.mean(z)  # Saved loss is the error rate
    
    total_loss  += lossnum
    if (numiter+1) % print_every == 0:
        print((numiter, "===="))
        print(target.cpu().numpy()[-10:])   # Target pattern to be reconstructed
        print(inputs.cpu().numpy()[numstep][0][-10:])  # Last input contains the degraded pattern fed to the network at test time
        print(y.data.cpu().numpy()[0][-10:])   # Final output of the network
        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", print_every, "iters: ", nowtime - previoustime)
        total_loss /= print_every
        all_losses.append(total_loss)
        print("Mean loss over last", print_every, "iters:", total_loss)
        print("")
        with open('output_simple_'+str(RNGSEED)+'.dat', 'wb') as fo:
            pickle.dump(net.w.data.cpu().numpy(), fo)
            pickle.dump(net.alpha.data.cpu().numpy(), fo)
            pickle.dump(y.data.cpu().numpy(), fo)  # The final y for this episode
            pickle.dump(all_losses, fo)
        with open('loss_simple_'+str(RNGSEED)+'.txt', 'w') as fo:
            for item in all_losses:
                fo.write("%s\n" % item)
        total_loss = 0



