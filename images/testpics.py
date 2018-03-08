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

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
plt.ion()


import pics_tosend as pics
from pics_tosend import Network

plt.figure()

# Note that this is a different file from the ones used in training
with open('./data_batch_5', 'rb') as fo:
    imagedict = pickle.load(fo, encoding='bytes')
imagedata = imagedict[b'data']

suffix='images_patternsize_1024_interpresdelay_2_nbpatterns_3_lr_0.0001_nbprescycles_3_homogenous_20_nbiter_100000_prestime_20_probadegrade_0.5_prestimetest_3_rngseed_0'
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
rngseed=4
np.random.seed(rngseed); random.seed(rngseed); torch.manual_seed(rngseed)

#print myall_losses

ttype = torch.cuda.FloatTensor # Must match the one in pics_eta.py
#ttype = torch.FloatTensor # Must match the one in pics_eta.py

net.w.data = torch.from_numpy(myw).type(ttype)
net.alpha.data = torch.from_numpy(myalpha).type(ttype)
net.eta.data = torch.from_numpy(myeta).type(ttype)
print(net.w.data[:10,:10])
print(net.eta.data)

NBPICS = 7
nn=1
for numpic in range(NBPICS):

    print("Pattern", numpic)

    inputsTensor, targetPattern = pics.generateInputsAndTarget(myparams, contiguousperturbation=True)

    y = net.initialZeroState()
    hebb = net.initialZeroHebb()
    #net.zeroDiagAlpha()

    for numstep in range(myparams['nbsteps']):
        y, hebb = net(Variable(inputsTensor[numstep], requires_grad=False), y, hebb)

   
    
    # All images could be  rotated 90deg. This allows us to display each set as a
    # vertical column by rotating the final image 90 degrees too.

    imagesize = int(np.sqrt(myparams['patternsize']))
    output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
    pattern1 = inputsTensor.cpu().numpy()[0][0][:-1].reshape((imagesize, imagesize))
    pattern2 = inputsTensor.cpu().numpy()[myparams['prestime']+myparams['interpresdelay']+1][0][:-1].reshape((imagesize, imagesize))
    pattern3 = inputsTensor.cpu().numpy()[2*(myparams['prestime']+myparams['interpresdelay'])+1][0][:-1].reshape((imagesize, imagesize))
    blankedpattern = inputsTensor.cpu().numpy()[-1][0][:-1].reshape((imagesize, imagesize))
    #output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize)).T
    #pattern1 = inputsTensor.cpu().numpy()[0][0][:-1].reshape((imagesize, imagesize)).T
    #pattern2 = inputsTensor.cpu().numpy()[myparams['prestime']+myparams['interpresdelay']+1][0][:-1].reshape((imagesize, imagesize)).T
    #pattern3 = inputsTensor.cpu().numpy()[2*(myparams['prestime']+myparams['interpresdelay'])+1][0][:-1].reshape((imagesize, imagesize)).T
    #blankedpattern = inputsTensor.cpu().numpy()[-1][0][:-1].reshape((imagesize, imagesize)).T

    plt.subplot(NBPICS,5,nn)
    plt.axis('off')
    plt.imshow(pattern1, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.subplot(NBPICS,5,nn+1)
    plt.axis('off')
    plt.imshow(pattern2, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.subplot(NBPICS,5,nn+2)
    plt.axis('off')
    plt.imshow(pattern3, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.subplot(NBPICS,5,nn+3)
    plt.axis('off')
    plt.imshow(blankedpattern, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.subplot(NBPICS,5,nn+4)
    plt.imshow(output, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')
    nn += 5

    td = targetPattern.cpu().numpy()
    yd = y.data.cpu().numpy()[0][:-1]
    absdiff = np.abs(td-yd)
    print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
    print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
    #print inputs[numstep]
plt.subplots_adjust(wspace=.1, hspace=.1)
