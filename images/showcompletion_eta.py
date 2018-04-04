# Old code to show the dynamics of pattern completion : show the product of the network at each time step
# Useful to understand how the network works (i.e. the need to clear up remnant activity from previous stimuli)
# May require adjustments to work (e.g. change file names, etc.)
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


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
import scipy
import scipy.misc
from torch import optim
import random
import sys
import pickle
import pdb
import time

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
plt.ion()


import pics_eta as pics
from pics_eta import Network

#plt.figure()

# Note that this is a different file from the ones used in training
with open('../data_batch_5', 'rb') as fo:
    imagedict = pickle.load(fo, encoding='bytes')
imagedata = imagedict[b'data']

#suffix = 'eta_prestime_20_probadegrade_0.5_interpresdelay_2_learningrate_0.0001_prestimetest_3_rngseed_0_nbiter_50000_nbprescycles_3_inputboost_1.0_eta_0.01_nbpatterns_3_patternsize_1024' # This one used for first draft of the paper, rngseed 4
#suffix = 'eta_inputboost_1.0_learningrate_0.0001_nbprescycles_3_interpresdelay_2_eta_0.01_rngseed_0_probadegrade_0.5_nbiter_150000_nbpatterns_3_prestimetest_3_patternsize_1024_prestime_20'
#suffix="eta_nbpatterns_3_inputboost_1.0_nbprescycles_3_prestime_20_prestimetest_5_interpresdelay_2_patternsize_1024_nbiter_50000_probadegrade_0.5_learningrate_0.0001_eta_0.01_rngseed_0"

suffix='etarefiner_eta_0.01_nbpatterns_3_interpresdelay_2_patternsize_1024_prestime_20_learningrate_1e-05_nbprescycles_3_rngseed_0_prestimetest_3_probadegrade_0.5_inputboost_1.0_nbiter_150000'


#fn = './tmp/results_'+suffix+'.dat'
fn = './results_'+suffix+'.dat'
with open(fn, 'rb') as fo:
    myw = pickle.load(fo)
    myalpha = pickle.load(fo)
    myeta = pickle.load(fo)
    myall_losses = pickle.load(fo)
    myparams = pickle.load(fo)

net = Network(myparams)

#np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
#rngseed=4
rngseed=7
np.random.seed(rngseed); random.seed(rngseed); torch.manual_seed(rngseed)

#print myall_losses

ttype = torch.cuda.FloatTensor # Must match the one in pics_eta.py
#ttype = torch.FloatTensor # Must match the one in pics_eta.py

net.w.data = torch.from_numpy(myw).type(ttype)
net.alpha.data = torch.from_numpy(myalpha).type(ttype)
net.eta.data = torch.from_numpy(myeta).type(ttype)
print(net.w.data[:10,:10])
print(net.eta.data)

NBPICS = 10
nn=1

imagesize = int(np.sqrt(myparams['patternsize']))
outputs={}
plt.figure()
FILLINGSTEPS = myparams['prestimetest'] + myparams['interpresdelay'] + 1

for numpic in range(NBPICS):

    print("Pattern", numpic)

    z = np.random.rand()
    z = np.random.rand()

    inputsTensor, targetPattern = pics.generateInputsAndTarget(myparams, contiguousperturbation=True)

    y = net.initialZeroState()
    hebb = net.initialZeroHebb()
    net.zeroDiagAlpha()

    for numstep in range(myparams['nbsteps']):
        y, hebb = net(Variable(inputsTensor[numstep], requires_grad=False), y, hebb)
        if numstep >= myparams['nbsteps'] - FILLINGSTEPS:
            output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
            #output = scipy.misc.imresize(output, 4.0)
            plt.subplot(NBPICS, FILLINGSTEPS, nn)
            plt.axis('off')
            plt.imshow(output, cmap='gray', vmin=-1.0, vmax=1.0)
            nn += 1
            #scipy.misc.imsave('pic'+str(numpic)+'_'+str(numstep)+'.png', output)


plt.show(block=True)
   
    
    # All images could be  rotated 90deg. This allows us to display each set as a
    # vertical column by rotating the final image 90 degrees too.

    #output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
    #pattern1 = inputsTensor.cpu().numpy()[0][0][:-1].reshape((imagesize, imagesize))
    #pattern2 = inputsTensor.cpu().numpy()[myparams['prestime']+myparams['interpresdelay']+1][0][:-1].reshape((imagesize, imagesize))
    #pattern3 = inputsTensor.cpu().numpy()[2*(myparams['prestime']+myparams['interpresdelay'])+1][0][:-1].reshape((imagesize, imagesize))
    #blankedpattern = inputsTensor.cpu().numpy()[-1][0][:-1].reshape((imagesize, imagesize))

    #plt.subplot(NBPICS,5,nn)
    #plt.axis('off')
    #plt.imshow(pattern1, cmap='gray', vmin=-1.0, vmax=1.0)
    #plt.subplot(NBPICS,5,nn+1)
    #plt.axis('off')
    #plt.imshow(pattern2, cmap='gray', vmin=-1.0, vmax=1.0)
    #plt.subplot(NBPICS,5,nn+2)
    #plt.axis('off')
    #plt.imshow(pattern3, cmap='gray', vmin=-1.0, vmax=1.0)
    #plt.subplot(NBPICS,5,nn+3)
    #plt.axis('off')
    #plt.imshow(blankedpattern, cmap='gray', vmin=-1.0, vmax=1.0)
    #plt.subplot(NBPICS,5,nn+4)
    #plt.imshow(output, cmap='gray', vmin=-1.0, vmax=1.0)
    #plt.axis('off')
    #nn += 5

    #td = targetPattern.cpu().numpy()
    #yd = y.data.cpu().numpy()[0][:-1]
    #absdiff = np.abs(td-yd)
    #print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
    #print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
    ##print inputs[numstep]
#plt.subplots_adjust(wspace=.1, hspace=.1)
