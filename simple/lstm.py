# Memorization of two 50-bit binary patterns per episode, with LSTMs. Takes a very long time to learn the task, and even then imperfectly. 2050 neurons (fewer neurons = worse performance).
#
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

# Uber-only (comment out if not at Uber)
import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs

# Parsing command-line arguments
params = {}; params['rngseed'] = 0
parser = argparse.ArgumentParser()
parser.add_argument("--rngseed", type=int, help="random seed", default=0)
parser.add_argument("--nbiter", type=int, help="number of episodes", default=2000)
parser.add_argument("--clamp", type=int, help="whether inputs are clamping (1) or not (0)", default=1)
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
NBHIDDENNEUR = PATTERNSIZE + params['nbaddneurons'] + 1  # NbNeur = Pattern Size + additional neurons + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
ETA = .01               # The "learning rate" of plastic connections; not used for LSTMs
ADAMLEARNINGRATE = params['lr']

PROBADEGRADE = .5       # Proportion of bits to zero out in the target pattern at test time
CLAMPING = params['clamp']
NBPATTERNS = params['nbpatterns'] # The number of patterns to learn in each episode
NBPRESCYCLES = params['nbprescycles']        # Number of times each pattern is to be presented
PRESTIME = params['prestime'] # Number of time steps for each presentation
PRESTIMETEST = PRESTIME        # Same thing but for the final test pattern
INTERPRESDELAY = params['interpresdelay']      # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode

RNGSEED = params['rngseed']

#PATTERNSIZE = 50 
#
## Note: For LSTM, there are PATTERNSIZE input and output neurons, and NBHIDDENNEUR neurons in the hidden recurrent layer
##NBNEUR = PATTERNSIZE  # NbNeur = Pattern Size + 1 "bias", fixed-output neuron (bias neuron not needed for this task, but included for completeness)
#NBHIDDENNEUR = 2000  #  1000 takes longer 
#
##ETA = .01               # The "learning rate" of plastic connections. Not used for LSTMs.
#ADAMLEARNINGRATE = 3e-5 # 1e-4  # 3e-5 works better in the long run. 1e-4 OK. 3e-4 fails.
#RNGSEED = 0
#
#PROBADEGRADE = .5       # Proportion of bits to zero out in the target pattern at test time
#NBPATTERNS = 2          # The number of patterns to learn in each episode
#NBPRESCYCLES = 1        # Number of times each pattern is to be presented
#PRESTIME = 3            # Number of time steps for each presentation
#PRESTIMETEST = 3        # Same thing but for the final test pattern
#INTERPRESDELAY = 1      # Duration of zero-input interval between presentations
#NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode

#ttype = torch.FloatTensor;
ttype = torch.cuda.FloatTensor;

# Generate the full list of inputs for an episode. The inputs are returned as a PyTorch tensor of shape NbSteps x 1 x NbNeur
def generateInputsAndTarget():
    #inputT = np.zeros((NBSTEPS, 1, NBNEUR)) #inputTensor, initially in numpy format...
    inputT = np.zeros((NBSTEPS, 1, PATTERNSIZE)) #inputTensor, initially in numpy format...

    # Create the random patterns to be memorized in an episode
    seedp = np.ones(PATTERNSIZE); seedp[:PATTERNSIZE//2] = -1
    patterns=[]
    for nump in range(NBPATTERNS):
        p = np.random.permutation(seedp)
        patterns.append(p)

    # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
    # Choosing the test pattern, partially zero'ed out, that the network will have to complete
    

    testpattern = random.choice(patterns).copy()
    #testpattern = patterns[1].copy()
   
    
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
        #inputT[nn][0][-1] = 1.0  # Bias neuron.
        inputT[nn] *= 100.0       # Strengthen inputs
    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(testpattern).type(ttype)

    return inputT, target






class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        self.lstm = torch.nn.LSTM(PATTERNSIZE, NBHIDDENNEUR).cuda() #input size, hidden size
        self.hidden = self.initialZeroState() # Note that the "hidden state" is a tuple (hidden state, cells state)


    def forward(self, inputs,):
        # Run the network over entire sequence of inputs
        self.hidden = self.initialZeroState()
        if CLAMPING:
            # This code allows us to make the inputs on the LSTM "clamping",
            # i.e. neurons that receive an input have their output clamped at
            # this value, to make it similar to the RNN architectures.
            #
            # Note that you get worse results if you don't use it ! ("CLAMPING = 0" above) (clamping automatically reduces chance error to ~.25, since all input bits are always correct)
            #
            #self.lstm.weight_hh_l0.data.fill_(0)
            #self.lstm.weight_ih_l0.data.fill_(0)
            self.lstm.bias_hh_l0.data.fill_(0)
            #self.lstm.bias_ih_l0.data.fill_(0)
            for ii in range(PATTERNSIZE):
                self.lstm.weight_ih_l0.data[2*NBHIDDENNEUR + ii].fill_(0)
                self.lstm.weight_ih_l0.data[2*NBHIDDENNEUR + ii][ii] = 10.0  # Trick to make inputs clamping on the cells, for fair comparison (need to also set input gates...)
                self.lstm.bias_ih_l0.data[0*NBHIDDENNEUR+ ii]= 10.0 # bias to input gate
                self.lstm.bias_ih_l0.data[1*NBHIDDENNEUR+ ii]= -1000.0 # bias to forget gate (actually a persistence gate? - sigmoid, so to set it to 0, put a massive negative bias)
                self.lstm.bias_ih_l0.data[2*NBHIDDENNEUR+ ii]= 0 # bias to cell gate
                self.lstm.bias_ih_l0.data[3*NBHIDDENNEUR+ ii]= 10.0 # bias to output gate; sigmoid
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        #o = self.h2o(lstm_out) #.view(NBSTEPS, -1))
        #outputz = F.tanh(o)
        outputz = lstm_out
        return outputz


        #yout = F.tanh( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
        #hebb = (1 - ETA) * hebb + ETA * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] # bmm used to implement outer product with the help of unsqueeze (i.e. added empty dimensions)
        #return yout, hebb

    def initialZeroState(self):
        return (Variable(torch.zeros(1, 1, NBHIDDENNEUR).type(ttype)),
                                Variable(torch.zeros(1, 1, NBHIDDENNEUR).type(ttype)))


if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))

np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)


net = NETWORK()
optimizer = torch.optim.Adam(net.parameters(), lr=ADAMLEARNINGRATE)
total_loss = 0.0; all_losses = []
print_every = 100
save_every = 1000
nowtime = time.time()

for numiter in range(params['nbiter']):
    
    optimizer.zero_grad()

    net.hidden = net.initialZeroState()

    # Generate the inputs and target pattern for this episode
    inputs, target = generateInputsAndTarget()

    # Run the episode!
    y = net(Variable(inputs, requires_grad=False))[-1][0]

    # Compute loss for this episode (last step only)
    loss = (y[:PATTERNSIZE] - Variable(target, requires_grad=False)).pow(2).sum()
    
    
    #pdb.set_trace()

    # Apply backpropagation to adapt basic weights and plasticity coefficients
    loss.backward()
    optimizer.step()

    # That's it for the actual algorithm.
    # Print statistics, save files
    #lossnum = loss.data[0]
    yo = y.data.cpu().numpy()[:PATTERNSIZE]
    to = target.cpu().numpy()
    z = (np.sign(yo) != np.sign(to))
    lossnum = np.mean(z)
    total_loss  += lossnum
    if (numiter+1) % print_every == 0:
        print((numiter, "===="))
        print(target.cpu().numpy()[:10])   # Target pattern to be reconstructed
        print(inputs.cpu().numpy()[-1][0][:10])  # Last input contains the degraded pattern fed to the network at test time
        print(y.data.cpu().numpy()[:10])   # Final output of the network
        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", print_every, "iters: ", nowtime - previoustime)
        total_loss /= print_every
        all_losses.append(total_loss)
        print("Mean loss over last", print_every, "iters:", total_loss)
        print("")
    if (numiter+1) % save_every == 0:
        fname = 'loss_binary_lstm_nbiter_'+str(params['nbiter'])+'_nbhneur_'+str(NBHIDDENNEUR)+'_clamp_'+str(CLAMPING)+'_lr_'+str(ADAMLEARNINGRATE)+'_prestime_'+str(PRESTIME)+'_ipd_'+str(INTERPRESDELAY)+'_rngseed_'+str(RNGSEED)+'.txt'
        with open(fname, 'w') as fo:
            for item in all_losses:
                fo.write("%s\n" % item)

        # Uber-only (comment out if not at Uber)
        if checkHdfs():
            print("Transfering to HDFS...")
            transferFileToHdfsDir(fname, '/ailabs/tmiconi/simple/')

        total_loss = 0



