# Make an animation from the activities of the network over time

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

np.set_printoptions(precision=3)


import pics_eta as pics
from pics_eta import Network

fig = plt.figure()
plt.axis('off')

# Note that this is a different file from the ones used in training
with open('./data_batch_5', 'rb') as fo:
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
#rngseed=18
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

NBPICS = 1 # 10 
nn=1

imagesize = int(np.sqrt(myparams['patternsize']))
outputs={}
FILLINGSTEPS = myparams['prestimetest'] + myparams['interpresdelay'] + 1





# Two ways to do it : show the full actual process, or show a "simnplified" version where you just show the three images and the pattern completion (slowed down)

SIMPLIFIED = 0

if SIMPLIFIED:

    for numpic in range(NBPICS):

        print("Pattern", numpic)

        z = np.random.rand()
        z = np.random.rand()

        inputsTensor, targetPattern = pics.generateInputsAndTarget(myparams, contiguousperturbation=True)

        y = net.initialZeroState()
        hebb = net.initialZeroHebb()
        net.zeroDiagAlpha()

        ax_imgs = []

        print("Running the episode...")
        for numstep in range(myparams['nbsteps']):
            y, hebb = net(Variable(inputsTensor[numstep], requires_grad=False), y, hebb)
            output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
            #output = scipy.misc.imresize(output, 4.0)
            #plt.subplot(NBPICS, FILLINGSTEPS, nn)
            #plt.axis('off')
            #plt.imshow(output, cmap='gray', vmin=-1.0, vmax=1.0)

            #if numstep == 1  or numstep == myparams['prestime'] + myparams['interpresdelay'] + 1 or  \
                    #numstep == 2 * (myparams['prestime'] + myparams['interpresdelay']) + 1 or \


            # Show the last set of 3 patterns, and the completion:
            if numstep ==  myparams['nbsteps'] - myparams['prestimetest'] - myparams['interpresdelay'] - 2 or \
                    numstep ==  myparams['nbsteps'] - myparams['prestimetest'] - (myparams['interpresdelay'] + myparams['prestime']) - myparams['interpresdelay'] - 2 or \
                    numstep ==  myparams['nbsteps'] - myparams['prestimetest'] - (myparams['interpresdelay'] + myparams['prestime']) *2 - myparams['interpresdelay'] - 2  or \
                    numstep >= myparams['nbsteps'] - myparams['prestimetest'] :
                if numstep == myparams['nbsteps'] - myparams['prestimetest'] :
                    output_half = output.copy()
                    output_half[16:,:] = 0      # NOTE: we are assuming that the grayed part will be the bottom one, which is only true for half the cases
                    a1 = plt.imshow(output_half, animated=True, cmap='gray', vmin=-1.0, vmax=1.0)
                else:
                    a1 = plt.imshow(output, animated=True, cmap='gray', vmin=-1.0, vmax=1.0)
                #a2 = plt.text(1, 1, str(numstep)+"/"+str(myparams['nbsteps']), fontsize=12, color='r')
                if numstep < myparams['nbsteps'] - myparams['prestimetest'] :
                        a3 = plt.text(1, 1,  "Pattern "+str(nn), fontsize=12, color='r')
                else:
                        a3 = plt.text(1, 1, "Pattern completion", fontsize=12, color='r')
                ax_imgs.append([a1, a3])  
                #ax_imgs.append([fullimg])  
                nn += 1
                #scipy.misc.imsave('pic'+str(numpic)+'_'+str(numstep)+'.png', output)


    #plt.show(block=True)
        print("Writing out the animation file")
        anim = animation.ArtistAnimation(fig, ax_imgs, repeat_delay=2000)  # repeat_delay is ignored...
        anim.save('anim_short_'+str(numpic)+'.gif', writer='imagemagick', fps=1)
       
        
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


else:
    for numpic in range(NBPICS):

        print("Pattern", numpic)

        z = np.random.rand()
        z = np.random.rand()

        inputsTensor, targetPattern = pics.generateInputsAndTarget(myparams, contiguousperturbation=True)

        y = net.initialZeroState()
        hebb = net.initialZeroHebb()
        net.zeroDiagAlpha()

        ax_imgs = []

        print("Running the episode...")
        for numstep in range(myparams['nbsteps']):
            y, hebb = net(Variable(inputsTensor[numstep], requires_grad=False), y, hebb)
            output = y.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
            #output = scipy.misc.imresize(output, 4.0)
            #plt.subplot(NBPICS, FILLINGSTEPS, nn)
            #plt.axis('off')
            #plt.imshow(output, cmap='gray', vmin=-1.0, vmax=1.0)
            a1 = plt.imshow(output, animated=True, cmap='gray', vmin=-1.0, vmax=1.0)
            a2 = plt.text(1, 1, str(numstep)+"/"+str(myparams['nbsteps']), fontsize=12, color='r')
            if numstep < myparams['nbsteps'] - myparams['prestimetest'] -  1:
                a3 = plt.text(14, 1,  "Pattern presentations", fontsize=12, color='r')
            else:
                a3 = plt.text(14, 1, "Pattern completion", fontsize=12, color='r')
            ax_imgs.append([a1, a2, a3])  
            #ax_imgs.append([fullimg])  
            nn += 1
            #scipy.misc.imsave('pic'+str(numpic)+'_'+str(numstep)+'.png', output)

        # Post-completion, keep the last image up a bit 
        for numstep_add in range(50):
            a1 = plt.imshow(output, animated=True, cmap='gray', vmin=-1.0, vmax=1.0)
            a2 = plt.text(1, 1, str(myparams['nbsteps'])+"/"+str(myparams['nbsteps']), fontsize=12, color='r')
            a3 = plt.text(14, 1, "Pattern completion", fontsize=12, color='r')
            ax_imgs.append([a1, a2, a3])  



    #plt.show(block=True)
        print("Writing out the animation file")
        anim = animation.ArtistAnimation(fig, ax_imgs, repeat_delay=2000)  # repeat_delay is ignored...
        anim.save('anim_full_'+str(numpic)+'.gif', writer='imagemagick', fps=10)
       
        
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
