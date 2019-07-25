# Stimulus-response task as described in Miconi et al. ICLR 2019.

# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

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
import platform
#import makemaze

import numpy as np
#import matplotlib.pyplot as plt
import glob

import modul




np.set_printoptions(precision=4)



ADDINPUT = 4 # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs


def train(paramdict):
    #params = dict(click.get_current_context().params)

    #params['inputsize'] =  RFSIZE * RFSIZE + ADDINPUT + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    #params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "SRB_"+"".join([str(x)+"_" if pair[0] != 'pe' and pair[0] != 'nbsteps' and pair[0] != 'rngseed' and pair[0] != 'save_every' and pair[0] != 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames
    print(suffix)

    #NBINPUTBITS = params['ni'] + 1 
    NBINPUTBITS = params['cs'] + 1 # The additional bit is for the response cue (i.e. the "Go" cue)
    params['outputsize'] =  2  # "response" and "no response"
    params['inputsize'] = NBINPUTBITS +  params['outputsize'] + ADDINPUT  # The total number of input bits is the size of inputs, plus the "response cue" input, plus the number of actions, plus the number of additional inputs

    # This doesn't work with our version of PyTorch
    #params['device'] = 'gpu'
    #device = torch.device("cuda:0" if self.params['device'] == 'gpu' else "cpu")
    BS = params['bs']

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)

    print("Initializing network")
    if params['type'] == 'modul':
        net = modul.RetroModulRNN(params)
    elif params['type'] == 'modplast':
        net = modul.SimpleModulRNN(params)
    elif params['type'] == 'plastic':
        net = modul.PlasticRNN(params)
    elif params['type'] == 'rnn':
        net = modul.NonPlasticRNN(params)
    else:
        raise ValueError("Network type unknown or not yet implemented: "+params['type'])

    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    #total_loss = 0.0
    print("Initializing optimizer")
    #optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'], weight_decay=params['l2'])
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=1.0*params['lr'], weight_decay=params['l2'])
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=params['eps'], weight_decay=params['l2'])
    #optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=1e-4, weight_decay=params['l2'])
    #optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

    #LABSIZE = params['lsize']
    #lab = np.ones((LABSIZE, LABSIZE))
    #CTR = LABSIZE // 2

    # Simple cross maze
    #lab[CTR, 1:LABSIZE-1] = 0
    #lab[1:LABSIZE-1, CTR] = 0


    # Double-T maze
    #lab[CTR, 1:LABSIZE-1] = 0
    #lab[1:LABSIZE-1, 1] = 0
    #lab[1:LABSIZE-1, LABSIZE - 2] = 0

    # Grid maze
    #lab[1:LABSIZE-1, 1:LABSIZE-1].fill(0)
    #for row in range(1, LABSIZE - 1):
    #    for col in range(1, LABSIZE - 1):
    #        if row % 2 == 0 and col % 2 == 0:
    #            lab[row, col] = 1
    #lab[CTR,CTR] = 0 # Not strictly necessary, but perhaps helps loclization by introducing a detectable irregularity in the center


    #LABSIZE = params['msize'] 
    #lab = np.ones((LABSIZE, LABSIZE))
    #CTR = LABSIZE // 2 


    ## Grid maze
    #lab[1:LABSIZE-1, 1:LABSIZE-1].fill(0)
    #for row in range(1, LABSIZE - 1):
    #    for col in range(1, LABSIZE - 1):
    #        if row % 2 == 0 and col % 2 == 0:
    #            lab[row, col] = 1
    #lab[CTR,CTR] = 0 # Not strictly necessary, but perhaps helps loclization by introducing a detectable irregularity in the center



    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    #meanreward = np.zeros((LABSIZE, LABSIZE))
    meanreward = np.zeros(params['ni'])
    meanrewardT = np.zeros((params['ni'], params['eplen']))

    nbtrials = [0]*BS
    totalnbtrials = 0
    nbtrialswithcc = 0


    print("Starting episodes!")

    for numepisode in range(params['nbiter']):

        PRINTTRACE = 0
        #if (numepisode+1) % (1 + params['pe']) == 0:
        if (numepisode+1) % (params['pe']) == 0:
            PRINTTRACE = 1

        #lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        #count = np.zeros((LABSIZE, LABSIZE))

        # # Select the reward location for this episode - not on a wall!
        # rposr = 0; rposc = 0
        # while lab[rposr, rposc] == 1:
        #     rposr = np.random.randint(1, LABSIZE - 1)
        #     rposc = np.random.randint(1, LABSIZE - 1)

        # # We always start the episode from the center (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        # posc = CTR
        # posr = CTR



        
        optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        if params['type'] != 'rnn':
            hebb = net.initialZeroHebb()
        if params['type'] == 'modul':
            et = net.initialZeroHebb() # Eligibility Trace is identical to Hebbian Trace in shape
            pw = net.initialZeroPlasticWeights()
        numactionchosen = 0


        # Generate the cues. Make sure they're all different (important when using very small cues for debugging, e.g. cs=2, ni=2)
        cuedata=[]
        for nb in range(BS):
            cuedata.append([])
            for ncue in range(params['ni']):
                assert len(cuedata[nb]) == ncue
                foundsame = 1
                cpt = 0
                while foundsame > 0 :
                    cpt += 1
                    if cpt > 10000:
                        # This should only occur with very weird parameters, e.g. cs=2, ni>4
                        raise ValueError("Could not generate a full list of different cues")
                    foundsame = 0
                    candidate = np.random.randint(2, size=params['cs']) * 2 - 1
                    for backtrace in range(ncue):
                        if np.array_equal(cuedata[nb][backtrace], candidate):
                            foundsame = 1

                cuedata[nb].append(candidate)


        reward = np.zeros(BS)
        sumreward = np.zeros(BS)
        rewards = []
        vs = []
        logprobs = []
        cues=[]
        for nb in range(BS):
            cues.append([])
        dist = 0
        numactionschosen = np.zeros(BS, dtype='int32')

        #reward = 0.0
        #rewards = []
        #vs = []
        #logprobs = []
        #sumreward = 0.0
        nbtrials = np.zeros(BS)
        nbrewardabletrials = np.zeros(BS)
        thistrialhascorrectcue = np.zeros(BS)
        triallength = np.zeros(BS, dtype='int32')
        correctcue = np.random.randint(params['ni'], size=BS)

        trialstep = np.zeros(BS, dtype='int32')  

        #print("EPISODE ", numepisode)
        for numstep in range(params['eplen']):

            #if params['clamp'] == 0:
            inputs = np.zeros((BS, params['inputsize']), dtype='float32') 
            #else:
            #    inputs = np.zeros((1, params['hs']), dtype='float32')

            for nb in range(BS):
            
                if trialstep[nb] == 0:
                    thistrialhascorrectcue[nb] = 0
                    # Trial length is randomly modulated for each trial; first time step always -1 (i.e. no input cue), last time step always response-cue (i.e. NBINPUTBITS-1).
                    #triallength = params['ni'] // 2  + 3 + np.random.randint(1 + params['ni'])  # 3 fixed-cue time steps (1st, last and next-to-last) + some random nb of no-cue time steps
                    triallength[nb] = params['ni'] // 2  + 3 + np.random.randint(params['ni'])  # 3 fixed-cue time steps (1st, last and next-to-last) + some random nb of no-cue time steps
                    
                    
                    
                    # In any trial, we only show half the cues (randomly chosen), once each:
                    mycues = [x for x in range(params['ni'])]
                    random.shuffle(mycues); mycues = mycues[:len(mycues) // 2]
                    # The rest is filled with no-input time steps (i.e. cue = -1), but also with the 3 fixed-cue steps (1st, last, next-to-last) 
                    for nc in range(triallength[nb] - 3  - len(mycues)):
                        mycues.append(-1)
                    random.shuffle(mycues)
                    mycues.insert(0, -1); mycues.append(params['ni']); mycues.append(-1)  # The first and last time step have no input (cue -1), the next-to-last has the response cue.
                    assert(len(mycues) == triallength[nb])
                    cues[nb] = mycues

            
                inputs[nb, :NBINPUTBITS] = 0
                if cues[nb][trialstep[nb]] > -1 and cues[nb][trialstep[nb]] < params['ni']:
                    #inputs[0, cues[trialstep]] = 1.0
                    inputs[nb, :NBINPUTBITS-1] = cuedata[nb][cues[nb][trialstep[nb]]][:]
                    if cues[nb][trialstep[nb]] == correctcue[nb]:
                        thistrialhascorrectcue[nb] = 1
                if cues[nb][trialstep[nb]] == params['ni']:
                    inputs[nb, NBINPUTBITS-1] = 1  # "Go" cue
                    

                inputs[nb, NBINPUTBITS + 0] = 1.0 # Bias neuron, probably not necessary
                inputs[nb,NBINPUTBITS +  1] = numstep / params['eplen']
                inputs[nb, NBINPUTBITS + 2] = 1.0 * reward[nb] # Reward from previous time step
                if numstep > 0:
                    inputs[nb, NBINPUTBITS + ADDINPUT + numactionschosen[nb]] = 1  # Previously chosen action

            inputsC = torch.from_numpy(inputs).cuda()
            # Might be better:
            #if rposr == posr and rposc = posc:
            #    inputs[0][-4] = 100.0
            #else:
            #    inputs[0][-4] = 0
            
            # Running the network

            ## Running the network
            if params['type'] == 'modplast':
                y, v, DAout, hidden, hebb = net(Variable(inputsC, requires_grad=False), hidden, hebb)  # y  should output raw scores, not probas
            elif params['type'] == 'modul':
                y, v, DAout, hidden, hebb, et, pw  = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)  # y  should output raw scores, not probas
            elif params['type'] == 'plastic':
                y, v, hidden, hebb = net(Variable(inputsC, requires_grad=False), hidden, hebb)  # y  should output raw scores, not probas
            elif params['type'] == 'rnn':
                y, v, hidden = net(Variable(inputsC, requires_grad=False), hidden)  # y  should output raw scores, not probas
            else:
                raise ValueError("Network type unknown or not yet implemented!")



            y = F.softmax(y, dim=1)
            # Must convert y to probas to use this !
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()  
            logprobs.append(distrib.log_prob(actionschosen))
            numactionschosen = actionschosen.data.cpu().numpy()    # Turn to scalar

            if PRINTTRACE:
                print("Step ", numstep, " Inputs (1st in batch): ", inputs[0,:params['inputsize']], " - Outputs(0): ", y.data.cpu().numpy()[0,:], " - action chosen(0): ", numactionschosen[0],
                        "TrialLen(0):", triallength[0], "trialstep(0):", trialstep[0], "TTHCC(0): ", thistrialhascorrectcue[0], " -Reward (previous step): ", reward[0], ", cues(0):", cues[0], ", cc(0):", correctcue[0])

                #print("Step ", numstep, " Inputs: ", inputs[0,:params['inputsize']], " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
                #        " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), "TrialLen:", triallength, "trialstep:", trialstep, "TTHCC: ", thistrialhascorrectcue, " -Reward (previous step): ", reward, ", cues:", cues, ", cc:", correctcue)

            reward = np.zeros(BS, dtype='float32')

            for nb in range(BS):
                if numactionschosen[nb] == 1:
                    # Small penalty for any non-rest action taken
                    reward[nb]  -= params['wp']
            
            
            ### DEBUGGING
            ## Easiest possible episode-dependent response (i.e. the easiest
            ## possible problem that actually require meta-learning, with ni=2)
            ## This one works pretty wel... But harder ones don't work well!
            #if numactionchosen == correctcue :
            #        reward = params['rew']
            #else:
            #        reward = -params['rew']


                trialstep[nb] += 1
                if trialstep[nb] == triallength[nb] - 1:
                    # This was the next-to-last step of the trial (and we showed the response signal, unless it was the first few steps in episode). 
                    assert(cues[nb][trialstep[nb] - 1] == params['ni'] or numstep < 2)
                    # We must deliver reward (which will be perceived by the agent at the next step), positive or negative, depending on response
                    if thistrialhascorrectcue[nb] and numactionschosen[nb] == 1:
                        reward[nb] += params['rew']
                    elif (not thistrialhascorrectcue[nb]) and numactionschosen[nb] == 0:
                        reward[nb] += params['rew']
                    else:
                        reward[nb] -= params['rew']

                    if np.random.rand() < params['pf']:
                        reward[nb] = -reward[nb]
                
                if trialstep[nb] == triallength[nb]:
                    # This was the last step of the trial (and we showed no input)
                    assert(cues[nb][trialstep[nb] - 1] == -1 or numstep < 2)
                    nbtrials[nb] += 1
                    totalnbtrials += 1
                    if thistrialhascorrectcue[nb]:
                        nbtrialswithcc += 1
                        #nbrewardabletrials += 1 
                    # Trial is dead, long live trial
                    trialstep[nb] = 0

                    # We initialize the hidden state between trials!
                    #if params['is'] == 1:
                    #    hidden = net.initialZeroState()



            rewards.append(reward)
            vs.append(v)
            sumreward += reward



            #if params['alg'] in ['A3C' , 'REIE' , 'REIT']:
            
            loss += (params['bent'] * y.pow(2).sum() / BS )   # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution, so we use this instead.

            

            ##if PRINTTRACE:
            ##    print("Probabilities:", y.data.cpu().numpy(), "Picked action:", numactionchosen, ", got reward", reward)
        
        R = Variable(torch.zeros(BS).cuda(), requires_grad=False)
        gammaR = params['gr']
        for numstepb in reversed(range(params['eplen'])) :
            R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]).cuda(), requires_grad=False)
            ctrR = R - vs[numstepb][0]
            lossv += ctrR.pow(2).sum() / BS
            loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BS  # Need to check if detach() is OK
            #pdb.set_trace()


        # Episode is done, now let's do the actual computations
        #gammaR = params['gr']
        #if params['alg'] == 'A3C':
        #    R = 0
        #    for numstepb in reversed(range(params['eplen'])) :
        #        R = gammaR * R + rewards[numstepb]
        #        lossv += (vs[numstepb][0] - R).pow(2)
        #        loss -= logprobs[numstepb] * (R - vs[numstepb].data[0][0])  # Not sure if the "data" is needed... put it b/c of worry about weird gradient flows
        #    loss += params['bv'] * lossv

        #elif params['alg'] in ['REI', 'REIE']:
        #    R = sumreward
        #    baseline = meanreward[correctcue]
        #    for numstepb in reversed(range(params['eplen'])) :
        #        loss -= logprobs[numstepb] * (R - baseline)
        #elif params['alg'] == 'REIT':
        #    R = 0
        #    for numstepb in reversed(range(params['eplen'])) :
        #        R = gammaR * R + rewards[numstepb]
        #        loss -= logprobs[numstepb] * (R - meanrewardT[correctcue, numstepb])
        #else:
        #    raise ValueError("Must select algo type")
        #elif params['alg'] == 'REINOB':
        #    R = sumreward
        #    for numstepb in reversed(range(params['eplen'])) :
        #        loss -= logprobs[numstepb] * R
        #elif params['alg'] == 'REITMP':
        #    R = 0
        #    for numstepb in reversed(range(params['eplen'])) :
        #        R = gammaR * R + rewards[numstepb]
        #        loss -= logprobs[numstepb] * R

        #else:
        #    raise ValueError("Which algo?")

        #meanreward[correctcue] = (1.0 - params['nu']) * meanreward[correctcue] + params['nu'] * sumreward
        ##meanreward[rposr, rposc] = (1.0 - params['nu']) * meanreward[rposr, rposc] + params['nu'] * sumreward
        #R = 0
        #for numstepb in reversed(range(params['eplen'])) :
        #    R = gammaR * R + rewards[numstepb]
        #    meanrewardT[correctcue, numstepb] = (1.0 - params['nu']) * meanrewardT[correctcue, numstepb] + params['nu'] * R

        loss += params['blossv'] * lossv
        loss /= params['eplen']

        if PRINTTRACE:
            #if params['alg'] == 'A3C':
            print("lossv: ", float(lossv))
            #elif params['alg'] in ['REI', 'REIE', 'REIT']:
            #    print("meanreward baselines: ", [meanreward[x] for x in range(params['ni'])])
            print ("Total reward for this episode(0):", sumreward[0], "Prop. of trials w/ rewarded cue:", (nbtrialswithcc / totalnbtrials))
            print("Nb trials for this episode(0):", nbtrials[0], "[2]:",nbtrials[2]," Total Nb of trials:", totalnbtrials)

        #if params['squash'] == 1:
        #    if sumreward < 0:
        #        sumreward = -np.sqrt(-sumreward)
        #    else:
        #        sumreward = np.sqrt(sumreward)
        #elif params['squash'] == 0:
        #    pass
        #else:
        #    raise ValueError("Incorrect value for squash parameter")

        #loss *= sumreward

        #for p in net.parameters():
        #    p.grad.data.clamp_(-params['clamp'], params['clamp'])
        loss.backward()
        all_grad_norms.append(torch.nn.utils.clip_grad_norm(net.parameters(), params['gc']))
        if numepisode > 100:  # Burn-in period for meanreward
            optimizer.step()


        #print(sumreward)
        lossnum = float(loss)
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())
        #all_total_rewards.append(sumreward[0])
            #all_losses_v.append(lossv.data[0])
        #total_loss  += lossnum


        if (numepisode+1) % params['pe'] == 0:

            print(numepisode, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward: ", np.sum(all_total_rewards[-params['pe']:])/ params['pe'])
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            if params['type'] == 'plastic' or params['type'] == 'lstmplastic':
                print("ETA: ", float(net.eta), "alpha[0,1]: ", net.alpha.data.cpu().numpy()[0,1], "w[0,1]: ", net.w.data.cpu().numpy()[0,1] )
            elif params['type'] == 'modul' or params['type'] == 'modul2':
                print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy(), " mean-abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())))
            elif params['type'] == 'rnn':
                print("w[0,1]: ", net.w.data.cpu().numpy()[0,1] )

        if (numepisode+1) % params['save_every'] == 0:
            print("Saving files...")
#            lossbetweensaves /= params['save_every']
#            print("Average loss over the last", params['save_every'], "episodes:", lossbetweensaves)
#            print("Alternative computation (should be equal):", np.mean(all_losses_objective[-params['save_every']:]))
            losslast100 = np.mean(all_losses_objective[-100:])
            print("Average loss over the last 100 episodes:", losslast100)
#            # Instability detection; necessary for SELUs, which seem to be divergence-prone
#            # Note that if we are unlucky enough to have diverged within the last 100 timesteps, this may not save us.
#            if losslast100 > 2 * lossbetweensavesprev:
#                print("We have diverged ! Restoring last savepoint!")
#                net.load_state_dict(torch.load('./torchmodel_'+suffix + '.txt'))
#            else:
            print("Saving local files...")
            #with open('params_'+suffix+'.dat', 'wb') as fo:
            #        #pickle.dump(net.w.data.cpu().numpy(), fo)
            #        #pickle.dump(net.alpha.data.cpu().numpy(), fo)
            #        #pickle.dump(net.eta.data.cpu().numpy(), fo)
            #        #pickle.dump(all_losses, fo)
            #        pickle.dump(params, fo)
            #with open('loss_'+suffix+'.txt', 'w') as thefile:
            #    for item in all_losses_objective:
            #            thefile.write("%s\n" % item)
            #with open('lossv_'+suffix+'.txt', 'w') as thefile:
            #    for item in all_losses_v:
            #            thefile.write("%s\n" % item)
            #with open('grads_'+suffix+'.txt', 'w') as thefile:
            #    for item in all_grad_norms[::10]:
            #            thefile.write("%s\n" % item)
            with open('loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                        thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            with open('params_'+suffix+'.dat', 'wb') as fo:
                pickle.dump(params, fo)
            print("Saving HDFS files...")
            if os.path.isdir('/mnt/share/tmiconi'):
                print("Transferring to NFS storage...")
                for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
                    result = os.system(
                        'cp {} {}'.format(fn, '/mnt/share/tmiconi/3level/'+fn))
                print("Done!")
#            lossbetweensavesprev = lossbetweensaves
#            lossbetweensaves = 0
#            sys.stdout.flush()
#            sys.stderr.flush()



if __name__ == "__main__":
#defaultParams = {
#    'type' : 'lstm',
#    'seqlen' : 200,
#    'hs': 500,
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
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    #parser.add_argument("--clamp", type=float, help="maximum (absolute value) gradient for clamping", default=1000000.0)
    #parser.add_argument("--wp", type=float, help="wall penalty (reward decrement for hitting a wall)", default=0.1)
    parser.add_argument("--rew", type=float, help="reward value (reward increment for taking correct action after correct stimulus)", default=1.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.0)
    #parser.add_argument("--pen", type=float, help="penalty value (reward decrement for taking any non-rest action)", default=.2)
    #parser.add_argument("--exprew", type=float, help="reward value (reward increment for hitting reward location)", default=.0)
    parser.add_argument("--bent", type=float, help="coefficient for the entropy reward (really Simpson index concentration measure)", default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    #parser.add_argument("--probarev", type=float, help="probability of reversal (random change) in desired stimulus-response, per time step", default=0.0)
    parser.add_argument("--bv", type=float, help="coefficient for value prediction loss", default=.1)
    #parser.add_argument("--lsize", type=int, help="size of the labyrinth; must be odd", default=7)
    #parser.add_argument("--randstart", type=int, help="when hitting reward, should we teleport to random location (1) or center (0)?", default=0)
    #parser.add_argument("--rp", type=int, help="whether the reward should be on the periphery", default=0)
    #parser.add_argument("--squash", type=int, help="squash reward through signed sqrt (1 or 0)", default=0)
    #parser.add_argument("--nbarms", type=int, help="number of arms", default=2)
    #parser.add_argument("--nbseq", type=int, help="number of sequences between reinitializations of hidden/Hebbian state and position", default=3)
    #parser.add_argument("--activ", help="activ function ('tanh' or 'selu')", default='tanh')
    parser.add_argument("--alg", help="meta-learning algorithm (A3C or REI or REIE or REIT)", default='REIT')
    parser.add_argument("--rule", help="learning rule ('hebb' or 'oja')", default='hebb')
    parser.add_argument("--type", help="network type ('lstm' or 'rnn' or 'plastic')", default='modul')
    #parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=9)
    parser.add_argument("--da", help="transformation function of DA signal (tanh or sig or lin)", default='tanh')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--fm", type=int, help="if using neuromodulation, do we modulate the whole network (1) or just half (0) ?", default=1)
    #parser.add_argument("--na", type=int, help="number of actions (excluding \"rest\" action)", default=2)
    parser.add_argument("--ni", type=int, help="number of different inputs", default=2)
    parser.add_argument("--nu", type=float, help="REINFORCE baseline time constant", default=.1)
    #parser.add_argument("--samestep", type=int, help="compare stimulus and response in the same step (1) or from successive steps (0) ?", default=0)
    #parser.add_argument("--nbin", type=int, help="number of possible inputs stimulis", default=4)
    #parser.add_argument("--modhalf", type=int, help="which half of the recurrent netowkr receives modulation (1 or 2)", default=1)
    #parser.add_argument("--nbac", type=int, help="number of possible non-rest actions", default=4)
    #parser.add_argument("--rsp", type=int, help="does the agent start each episode from random position (1) or center (0) ?", default=1)
    parser.add_argument("--addpw", type=int, help="are plastic weights purely additive (1) or forgetting (0) ?", default=2)
    parser.add_argument("--clamp", type=int, help="inputs clamped (1), fully clamped (2) or through linear layer (0) ?", default=0)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    #parser.add_argument("--exptime", type=int, help="exploration (no reward) time (must be < eplen)", default=0)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--is", type=int, help="do we initialize hidden state after each trial (1) or not (0) ?", default=0)
    parser.add_argument("--cs", type=int, help="cue size - number of bits for each cue", default=10)
    parser.add_argument("--pf", type=float, help="probability of flipping the reward (.5 = pure noise)", default=0)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=1e-5)
    parser.add_argument("--bs", type=int, help="batch size", default=1)
    parser.add_argument("--gc", type=float, help="gradient clipping", default=1000.0)
    parser.add_argument("--eps", type=float, help="epsilon for Adam optimizer", default=1e-6)
    #parser.add_argument("--steplr", type=int, help="duration of each step in the learning rate annealing schedule", default=100000000)
    #parser.add_argument("--gamma", type=float, help="learning rate annealing factor", default=0.3)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=200)
    parser.add_argument("--pe", type=int, help="'print every', number of cycles between successive printing of information", default=100)
    #parser.add_argument("--", type=int, help="", default=1e-4)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    #train()
    train(argdict)

