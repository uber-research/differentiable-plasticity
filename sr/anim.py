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

import modul
from modul import Network

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

    suffix = "modulmaze_"+"".join([str(x)+"_" if pair[0] != 'nbsteps' and pair[0] != 'rngseed' and pair[0] != 'save_every' and pair[0] != 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames


    #params['rngseed'] = 3
    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)
    
    net = Network(params)
    # YOU MAY NEED TO CHANGE THE DIRECTORY HERE:
    net.load_state_dict(torch.load('./tmp/torchmodel_'+suffix + '.dat'))


    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    LABSIZE = params['msize'] 
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2 

    # Grid maze
    lab[1:LABSIZE-1, 1:LABSIZE-1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    lab[CTR,CTR] = 0 # Not strictly necessary, but perhaps helps loclization by introducing a detectable irregularity in the center



    all_losses = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))

    pos = 0


    
    params['nbiter'] = 3
    ax_imgs = []
    
    for numiter in range(params['nbiter']):

        PRINTTRACE = 0
        #if (numiter+1) % (1 + params['print_every']) == 0:
        if (numiter+1) % (params['print_every']) == 0:
            PRINTTRACE = 1

        #lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        #count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        rposr = 0; rposc = 0
        while lab[rposr, rposc] == 1:
            rposr = np.random.randint(1, LABSIZE - 1)
            rposc = np.random.randint(1, LABSIZE - 1)

        # We always start the episode from the center (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        posc = CTR
        posr = CTR

        #optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()
        et = net.initialZeroHebb()
        pw = net.initialZeroPlasticWeights()
        numactionchosen = 0


        reward = 0.0
        rewards = []
        vs = []
        logprobs = []
        sumreward = 0.0
        dist = 0
        

        #print("EPISODE ", numiter)
        for numstep in range(params['eplen']):


            if params['clamp'] == 0:
                inputs = np.zeros((1, TOTALNBINPUTS), dtype='float32') 
            else:
                inputs = np.zeros((1, params['hs']), dtype='float32')
        
            labg = lab.copy()
            #labg[rposr, rposc] = -1  # The agent can see the reward if it falls within its RF
            inputs[0, 0:RFSIZE * RFSIZE] = labg[posr - RFSIZE//2:posr + RFSIZE//2 +1, posc - RFSIZE //2:posc + RFSIZE//2 +1].flatten() * 1.0
            
            # Previous chosen action
            inputs[0, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
            inputs[0, RFSIZE * RFSIZE +2] = numstep / params['eplen']
            inputs[0, RFSIZE * RFSIZE +3] = 1.0 * reward # Reward from previous time step
            inputs[0, RFSIZE * RFSIZE + ADDINPUT + numactionchosen] = 1
            inputsC = torch.from_numpy(inputs).cuda()

            ## Running the network
            y, v, hidden, hebb, et, pw = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)  # y  should output raw scores, not probas


            y = F.softmax(y, dim=1)
            # Must convert y to probas to use this !
            distrib = torch.distributions.Categorical(y)
            actionchosen = distrib.sample()  # sample() returns a Pytorch tensor of size 1; this is needed for the backprop below
            numactionchosen = actionchosen.data[0]    # Turn to scalar

            tgtposc = posc
            tgtposr = posr
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
            
            reward = 0.0
            if lab[tgtposr][tgtposc] == 1:
                # Hit wall!
                reward = -params['wp']
            else:
                dist += 1
                posc = tgtposc
                posr = tgtposr
            
            
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
            labg[rposr, rposc] = 2
            labg[posr, posc] = 3
            fullimg = plt.imshow(labg, animated=True)
            ax_imgs.append([fullimg])  


            # Did we hit the reward location ? Increase reward and teleport!
            # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move
            if rposr == posr and rposc == posc:
                reward += params['rew']
                if params['rsp'] == 1:
                    posr = np.random.randint(1, LABSIZE - 1)
                    posc = np.random.randint(1, LABSIZE - 1)
                    while lab[posr, posc] == 1:
                        posr = np.random.randint(1, LABSIZE - 1)
                        posc = np.random.randint(1, LABSIZE - 1)
                else:
                    posr = CTR
                    posc = CTR


            #if PRINTTRACE:
            #    #print("Step ", numstep, "- GI: ", goodinput, ", GA: ", goodaction, " Inputs: ", inputsN, " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
            #    #        " - inputthisstep:", inputthisstep, " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Rew: ", reward)
            #    print("Step ", numstep, " Inputs: ", inputs[0,:TOTALNBINPUTS], " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
            #            " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Reward (this step): ", reward)
            rewards.append(reward)
            vs.append(v)
            sumreward += reward



            logprobs.append(distrib.log_prob(actionchosen))

            #if params['algo'] == 'A3C':
            loss += params['bentropy'] * y.pow(2).sum()   # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution, so we use this instead.

            ##if PRINTTRACE:
            ##    print("Probabilities:", y.data.cpu().numpy(), "Picked action:", numactionchosen, ", got reward", reward)


        # Episode is done, now let's do the actual computations
        gammaR = params['gr']
        if True: #params['algo'] == 'A3C':
            R = 0
            for numstepb in reversed(range(params['eplen'])) :
                R = gammaR * R + rewards[numstepb]
                lossv += (vs[numstepb][0] - R).pow(2)
                loss -= logprobs[numstepb] * (R - vs[numstepb].data[0][0])  # Not sure if the "data" is needed... put it b/c of worry about weird gradient flows
            loss += params['blossv'] * lossv

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

        meanrewards[rposr, rposc] = (1.0 - params['nu']) * meanrewards[rposr, rposc] + params['nu'] * sumreward
        R = 0
        for numstepb in reversed(range(params['eplen'])) :
            R = gammaR * R + rewards[numstepb]
            meanrewardstmp[rposr, rposc, numstepb] = (1.0 - params['nu']) * meanrewardstmp[rposr, rposc, numstepb] + params['nu'] * R

        loss /= params['eplen']

        if True: #PRINTTRACE:
            if True: #params['algo'] == 'A3C':
                print("lossv: ", lossv.data.cpu().numpy()[0])
            print ("Total reward for this episode:", sumreward, "Dist:", dist)

        #if numiter > 100:  # Burn-in period for meanrewards
        #    loss.backward()
        #    optimizer.step()

        #torch.cuda.empty_cache()

        #print(sumreward)
        lossnum = loss.data[0]
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward)
            #all_losses_v.append(lossv.data[0])
        #total_loss  += lossnum


        if True: #PRINTTRACE:
            print("lossv: ", lossv.data.cpu().numpy()[0])
            print ("Total reward for this episode:", sumreward, "Dist:", dist)


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
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    train(argdict)

