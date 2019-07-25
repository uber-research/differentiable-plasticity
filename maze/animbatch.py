# This code produces animations showing the behavior of an agent for two successive episodes.

# Usage: python animbatch.py --file FILENAME  [--initialize [0/1]]

# FILENAME should be the params_XXX.dat produced by the meta-learning process
# (batch.py). Make sure that the torchmodel_xxx.dat file is in the same location.

# Optional argument initialize should be set to 1 if you want to ignore the
# trained parameters and reinitialize the network, equivalent to obtaining the
# "generation-0" network. 


import argparse
import pdb 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs
import platform

import batch
from batch import Network

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob





np.set_printoptions(precision=4)

ETA = .02  # Not used

ADDINPUT = 4 # 1 input for the previous reward, 1 input for numstep, 1 for whether currently on reward square, 1 "Bias" input

NBACTIONS = 4  # U, D, L, R

RFSIZE = 3 # Receptive Field

TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBACTIONS


fig = plt.figure()
plt.axis('off')

def train(paramdict):

    fname = paramdict['file']

    with open(fname, 'rb') as f:
        params = pickle.load(f)

    #params = dict(click.get_current_context().params)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode

    suffix = "btchFixmod_"+"".join([str(x)+"_" if pair[0] != 'nbsteps' and pair[0] != 'rngseed' and pair[0] != 'save_every' and pair[0] != 'test_every' and pair[0] != 'pe' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames
    #suffix = "modRPDT_"+"".join([str(x)+"_" if pair[0] != 'nbsteps' and pair[0] != 'rngseed' and pair[0] != 'save_every' and pair[0] != 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames
    print("Reconstructed suffix:", suffix)


    params['rsp'] = 1

    #params['rngseed'] = 3
    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)
    
    net = Network(params)
    # YOU MAY NEED TO CHANGE THE DIRECTORY HERE:
    if paramdict['initialize'] == 0:
        net.load_state_dict(torch.load('./tmp/torchmodel_'+suffix + '.dat'))


    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    BATCHSIZE = params['bs']

    LABSIZE = params['msize'] 
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2 

    # Simple cross maze
    #lab[CTR, 1:LABSIZE-1] = 0
    #lab[1:LABSIZE-1, CTR] = 0


    # Double-T maze
    #lab[CTR, 1:LABSIZE-1] = 0
    #lab[1:LABSIZE-1, 1] = 0
    #lab[1:LABSIZE-1, LABSIZE - 2] = 0

    # Grid maze
    lab[1:LABSIZE-1, 1:LABSIZE-1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR,CTR] = 0 



    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))


    pos = 0
    hidden = net.initialZeroState()
    hebb = net.initialZeroHebb()
    pw = net.initialZeroPlasticWeights()

    #celoss = torch.nn.CrossEntropyLoss() # For supervised learning - not used here




    
    params['nbiter'] = 3
    ax_imgs = []
    
    for numiter in range(params['nbiter']):

        PRINTTRACE = 0
        #if (numiter+1) % (1 + params['pe']) == 0:
        if (numiter+1) % (params['pe']) == 0:
            PRINTTRACE = 1

        #lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        #count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        posr = {}; posc = {}
        rposr = {}; rposc = {}
        for nb in range(BATCHSIZE):
            # Note: it doesn't matter if the reward is on the center (see below). All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0; myrposc = 0
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr; rposc[nb] = myrposc
            #print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR

        #optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()
        et = net.initialZeroHebb() # Eligibility Trace is identical to Hebbian Trace in shape
        pw = net.initialZeroPlasticWeights()
        numactionchosen = 0


        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        #reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)


        #print("EPISODE ", numiter)
        for numstep in range(params['eplen']):

            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32') 
        
            labg = lab.copy()
            #labg[rposr, rposc] = -1  # The agent can see the reward if it falls within its RF
            for nb in range(BATCHSIZE):
                inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE//2:posr[nb] + RFSIZE//2 +1, posc[nb] - RFSIZE //2:posc[nb] + RFSIZE//2 +1].flatten() * 1.0
                
                # Previous chosen action
                inputs[nb, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
                inputs[nb, RFSIZE * RFSIZE +2] = numstep / params['eplen']
                #inputs[0, RFSIZE * RFSIZE +3] = 1.0 * reward # Reward from previous time step
                inputs[nb, RFSIZE * RFSIZE +3] = 1.0 * reward[nb]
                inputs[nb, RFSIZE * RFSIZE + ADDINPUT + numactionschosen[nb]] = 1
                #inputs = 100.0 * inputs  # input boosting : Very bad with clamp=0
            
            inputsC = torch.from_numpy(inputs).cuda()
            # Might be better:
            #if rposr == posr and rposc = posc:
            #    inputs[0][-4] = 100.0
            #else:
            #    inputs[0][-4] = 0
            
            # Running the network

            ## Running the network
            y, v, hidden, hebb, et, pw = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)  # y  should output raw scores, not probas

            # For now:
            #numactionchosen = np.argmax(y.data[0])
            # But wait, this is bad, because the network needs to see the
            # reward signal to guide its own (within-episode) learning... and
            # argmax might not provide enough exploration for this!

            #ee = np.exp(y.data[0].cpu().numpy())
            #numactionchosen = np.random.choice(NBNONRESTACTIONS, p = ee / (1e-10 + np.sum(ee)))

            y = F.softmax(y, dim=1)
            # Must convert y to probas to use this !
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()  
            logprobs.append(distrib.log_prob(actionschosen))
            numactionschosen = actionschosen.data.cpu().numpy()    # Turn to scalar
            reward = np.zeros(BATCHSIZE, dtype='float32')
            #if numiter == 7 and numstep == 1:
            #    pdb.set_trace()


            for nb in range(BATCHSIZE):
                myreward = 0
                numactionchosen = numactionschosen[nb]

                tgtposc = posc[nb]
                tgtposr = posr[nb]
                if numactionchosen == 0:  # Up
                    tgtposr -= 1
                elif numactionchosen == 1:  # Down
                    tgtposr += 1
                elif numactionchosen == 2:  # Left
                    tgtposc -= 1
                elif numactionchosen == 3:  # Right
                    tgtposc += 1
                else:
                    raise ValueError("Wrong Action")
                
                reward[nb] = 0.0  # The reward for this step
                if lab[tgtposr][tgtposc] == 1:
                    reward[nb] -= params['wp']
                else:
                    #dist += 1
                    posc[nb] = tgtposc
                    posr[nb] = tgtposr

                # Did we hit the reward location ? Increase reward and teleport!
                # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move
                if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                    reward[nb] += params['rew']
                    posr[nb]= np.random.randint(1, LABSIZE - 1)
                    posc[nb] = np.random.randint(1, LABSIZE - 1)
                    while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                        posr[nb] = np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)

            rewards.append(reward)
            vs.append(v)
            sumreward += reward

            loss += ( params['bent'] * y.pow(2).sum() / BATCHSIZE )  # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution. Note: .2 may be too strong, .04 may be too weak. 
            #lossentmean  = .99 * lossentmean + .01 * ( params['bent'] * y.pow(2).sum() / BATCHSIZE ).data[0] # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution. Note: .2 may be too strong, .04 may be too weak. 


            if PRINTTRACE:
                #print("Step ", numstep, "- GI: ", goodinputs, ", GA: ", goodaction, " Inputs: ", inputsN, " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
                #        " - inputsthisstep:", inputsthisstep, " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Rew: ", reward)
                print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
                        " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Reward (this step, 1st in batch): ", reward[0])


            # Display the labyrinth

            #for numr in range(LABSIZE):
            #    s = ""
            #    for numc in range(LABSIZE):
            #        if posr == numr and posc == numc:
            #            s += "o"
            #        elif rposr == numr and rposc == numc:
            #            s += "X"
            #        elif lab[numr, numc] == 1:
            #            s += "#"
            #        else:
            #            s += " "
            #    print(s)
            #print("")
            #print("")

            labg = lab.copy()
            labg[rposr[0], rposc[0]] = 2
            labg[posr[0], posc[0]] = 3
            fullimg = plt.imshow(labg, animated=True)
            ax_imgs.append([fullimg])  




        # Episode is done, now let's do the actual computations

        R = Variable(torch.zeros(BATCHSIZE).cuda(), requires_grad=False)
        gammaR = params['gr']
        for numstepb in reversed(range(params['eplen'])) :
            R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]).cuda(), requires_grad=False)
            ctrR = R - vs[numstepb][0]
            lossv += ctrR.pow(2).sum() / BATCHSIZE
            loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  # Need to check if detach() is OK
            #pdb.set_trace()


        #elif params['algo'] == 'REI':
        #    R = sumreward
        #    baseline = meanrewards[rposr, rposc]
        #    for numstepb in reversed(range(params['eplen'])) :
        #        loss -= logprobs[numstepb] * (R - baseline)
        #elif params['algo'] == 'REINOB':
        #    R = sumreward
        #    for numstepb in reversed(range(params['eplen'])) :
        #        loss -= logprobs[numstepb] * R
        #elif params['algo'] == 'REITMP':
        #    R = 0
        #    for numstepb in reversed(range(params['eplen'])) :
        #        R = gammaR * R + rewards[numstepb]
        #        loss -= logprobs[numstepb] * R
        #elif params['algo'] == 'REITMPB':
        #    R = 0
        #    for numstepb in reversed(range(params['eplen'])) :
        #        R = gammaR * R + rewards[numstepb]
        #        loss -= logprobs[numstepb] * (R - meanrewardstmp[rposr, rposc, numstepb])

        #else:
        #    raise ValueError("Which algo?")

        #meanrewards[rposr, rposc] = (1.0 - params['nu']) * meanrewards[rposr, rposc] + params['nu'] * sumreward
        #R = 0
        #for numstepb in reversed(range(params['eplen'])) :
        #    R = gammaR * R + rewards[numstepb]
        #    meanrewardstmp[rposr, rposc, numstepb] = (1.0 - params['nu']) * meanrewardstmp[rposr, rposc, numstepb] + params['nu'] * R


        loss += params['blossv'] * lossv
        loss /= params['eplen']

        if True: #PRINTTRACE:
            if True: #params['algo'] == 'A3C':
                print("lossv: ", float(lossv))
            print ("Total reward for this episode:", sumreward, "Dist:", dist)

        #if numiter > 100:  # Burn-in period for meanrewards
        #    loss.backward()
        #    optimizer.step()

        #torch.cuda.empty_cache()


    print("Saving animation....")
    anim = animation.ArtistAnimation(fig, ax_imgs, interval=200)
    anim.save('anim.gif', writer='imagemagick', fps=10)



if __name__ == "__main__":
#defaultParams = {
#    'type' : 'lstm',
#    'seqlen' : 200,
#    'hiddensize': 500,
#    'activ': 'tanh',
#    'steplr': 10e9,  # By default, no change in the learning rate
#    'gamma': .5,  # The annealing factor of learning rate decay for Adam
#    'imagesize': 31,    
#    'nbiter': 30000,  
#    'lr': 1e-4,   
#    'test_every': 10,
#    'save_every': 3000,
#    'rngseed':0
#}
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="params file")
    parser.add_argument("--initialize", help="should we reinitialize the network (1) or keep the trained network (0)?", default=0)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    train(argdict)

