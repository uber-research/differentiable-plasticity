# Differentiable plasticity: binary pattern memorization and reconstruction
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# This more flexible implementation includes both plastic and non-plastic RNNs. LSTM code is sufficiently different that it makes more sense to put it in a different file.
# Also includes some Uber-specific stuff for file transfer. Commented out by default.


# Parameters optimized for non-plastic architectures (esp. LSTM): 
# --patternsize 50 --nbaddneurons 2000 --nbprescycles 1 --nbpatterns 2 --prestime 3 --interpresdelay 1 --nbiter 1000000 --lr 3e-5
# For comparing plastic and non-plastic, we use these for both (though the plastic architecture strongly prefers the default ones)
# Plastic networks can learn with lr=3e-4.
# The default parameters are those for the plastic RNN on the 1000-bit task (same as simple.py) 

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
import pickle
import pdb
import time

# Uber-only (comment out if not at Uber):
import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs


# Parsing command-line arguments
params = {}; params['rngseed'] = 0
parser = argparse.ArgumentParser()
parser.add_argument("--rngseed", type=int, help="random seed", default=0)
parser.add_argument("--nbiter", type=int, help="number of episodes", default=2000)
parser.add_argument("--nbaddneurons", type=int, help="number of additional neurons", default=0)
parser.add_argument("--lr", type=float, help="learning rate of Adam optimizer", default=3e-4)
parser.add_argument("--patternsize", type=int, help="size of the binary patterns", default=1000)
parser.add_argument("--nbpatterns", type=int, help="number of patterns to memorize", default=5)
parser.add_argument("--nbprescycles", type=int, help="number of presentation cycles", default=2)
parser.add_argument("--prestime", type=int, help="number of time steps for each pattern presentation", default=6)
parser.add_argument("--interpresdelay", type=int, help="number of time steps between each pattern presentation (with zero input)", default=4)
parser.add_argument("--type", help="network type ('plastic' or 'nonplastic')", default='plastic')
args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
params.update(argdict)

PATTERNSIZE = params['patternsize']
NBNEUR = PATTERNSIZE + params['nbaddneurons'] + 1  # NbNeur = Pattern Size + additional neurons + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
ETA = .01               # The "learning rate" of plastic connections
ADAMLEARNINGRATE = params['lr']

PROBADEGRADE = .5       # Proportion of bits to zero out in the target pattern at test time
NBPATTERNS = params['nbpatterns'] # The number of patterns to learn in each episode
NBPRESCYCLES = params['nbprescycles']        # Number of times each pattern is to be presented
PRESTIME = params['prestime'] # Number of time steps for each presentation
PRESTIMETEST = PRESTIME        # Same thing but for the final test pattern
INTERPRESDELAY = params['interpresdelay']      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode



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
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The eta coefficient is learned
        self.zeroDiagAlpha()  # No plastic autapses

    def forward(self, input, yin, hebb):
        # Run the network for one timestep
        if params['type'] == 'plastic':
            yout = F.tanh( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] # bmm used to implement outer product with the help of unsqueeze (i.e. added empty dimensions)
        elif params['type'] == 'nonplastic':
            yout = F.tanh( yin.mm(self.w) + input )
        else:
            raise ValueError("Wrong network type!")
        return yout, hebb

    def initialZeroState(self):
        return Variable(torch.zeros(1, NBNEUR).type(ttype))

    def initialZeroHebb(self):
        return Variable(torch.zeros(NBNEUR, NBNEUR).type(ttype))

    def zeroDiagAlpha(self):
        # Zero out the diagonal of the matrix of alpha coefficients: no plastic autapses
        self.alpha.data -= torch.diag(torch.diag(self.alpha.data))


np.set_printoptions(precision=3)
np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])




net = NETWORK()
optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=ADAMLEARNINGRATE)
total_loss = 0.0; all_losses = []
print_every = 100
save_every = 1000
nowtime = time.time()
suffix = "binary_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames

for numiter in range(params['nbiter']):
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
    net.zeroDiagAlpha()  # Make sure that no plastic autapses

    # That's it for the actual algorithm.
    # Print statistics, save files
    #lossnum = loss.data[0]     # Saved loss is the actual training loss (MSE)
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
    if (numiter+1) % save_every == 0:
        with open('outputs_'+suffix+'.dat', 'wb') as fo:
            pickle.dump(net.w.data.cpu().numpy(), fo)
            pickle.dump(net.alpha.data.cpu().numpy(), fo)
            pickle.dump(y.data.cpu().numpy(), fo)  # The final y for this episode
            pickle.dump(all_losses, fo)
        with open('loss_'+suffix+'.txt', 'w') as fo:
            for item in all_losses:
                fo.write("%s\n" % item)
        # Uber-only
        if checkHdfs():
            print("Transfering to HDFS...")
            transferFileToHdfsDir('loss_'+suffix+'.txt', '/ailabs/tmiconi/simple/')
            #transferFileToHdfsDir('results_simple_'+str(params['rngseed'])+'.dat', '/ailabs/tmiconi/exp/')

        total_loss = 0



