 
# Differentiable plasticity: maze exploration task.
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



# NOTE: Do NOT use the 'lstmplastic' in this code. Instead, look at the
# awd-lstm-lm directory in this repo for properly implemented plastic LSTMs.


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

# Uber-only:
import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs
 
import numpy as np
#import matplotlib.pyplot as plt
import glob
 
 
 
np.set_printoptions(precision=4)
 
ETA = .02  # Not used
 
ADDINPUT = 4 # 1 input for the previous reward, 1 input for numstep, 1 for whether currently on reward square, 1 "Bias" input
 
NBACTIONS = 4  # U, D, L, R
 
RFSIZE = 3 # Receptive field size
 
TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBACTIONS
 

##ttype = torch.FloatTensor;    # For CPU
ttype = torch.cuda.FloatTensor; # Gor GPU


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        self.rule = params['rule']
        self.type = params['type']
        self.softmax= torch.nn.functional.softmax
        if params['activ'] == 'tanh':
            self.activ = F.tanh
        elif params['activ'] == 'selu':
            self.activ = F.selu
        else:
            raise ValueError('Must choose an activ function')
        if params['type'] == 'lstm':
            self.lstm = torch.nn.LSTM(TOTALNBINPUTS, params['hiddensize']).cuda()
        elif params['type'] == 'rnn':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
        elif params['type'] == 'homo':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
            self.alpha = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True) # Homogenous plasticity: everyone has the same alpha
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)   # Everyone has the same eta
        elif params['type'] == 'plastic':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta


        elif params['type'] == 'lstmplastic':   # LSTM with plastic connections. HIGHLY EXPERIMENTAL, NOT DEBUGGED - see awd-lstm-lm directory in this repo instead.
            self.h2f = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.h2i = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.h2opt = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            
            # Plasticity only in the recurrent connections, h to c.
            #self.h2c = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()  # This is replaced by the plastic connection matrices below
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hiddensize'], params['hiddensize'])).cuda(), requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta

            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
        elif params['type'] == 'lstmmanual':   # An LSTM implemented "by hand", to ensure maximum simlarity with the plastic LSTM
            self.h2f = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.h2i = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.h2opt = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.h2c = torch.nn.Linear(params['hiddensize'], params['hiddensize']).cuda()
            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hiddensize']).cuda()
            ##fgt = F.sigmoid(self.x2f(input) + self.h2f(hidden[0]))
            ##ipt = F.sigmoid(self.x2i(input) + self.h2i(hidden[0]))
            ##opt = F.sigmoid(self.x2o(input) + self.h2o(hidden[0]))
            ##cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(input) + self.h2c(hidden[0])))
            ##h = torch.mul(opt, cell)
            ##hidden = (h, cell)
        else:
            raise ValueError("Which network type?")
        self.h2o = torch.nn.Linear(params['hiddensize'], NBACTIONS).cuda()  # From hidden to action output
        self.h2v = torch.nn.Linear(params['hiddensize'], 1).cuda()          # From hidden to value prediction (for A3C)
        self.params = params
        
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

    def forward(self, input, hidden, hebb):
        if self.type == 'lstm':
            hactiv, hidden = self.lstm(input.view(1, 1, -1), hidden)  # hactiv is just the h. hidden is the h and the cell state, in a tuple
            hactiv = hactiv.view(1, -1)

        elif self.type == 'rnn':
            hactiv = self.activ(self.i2h(input) + hidden.mm(self.w))
            hidden = hactiv

        # Draft for a "manual" lstm:
        elif self.type== 'lstmmanual':
            # hidden[0] is the previous h state. hidden[1] is the previous c state
            fgt = F.sigmoid(self.x2f(input) + self.h2f(hidden[0]))
            ipt = F.sigmoid(self.x2i(input) + self.h2i(hidden[0]))
            opt = F.sigmoid(self.x2opt(input) + self.h2opt(hidden[0]))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(input) + self.h2c(hidden[0])))
            hactiv = torch.mul(opt, F.tanh(cell))
            #pdb.set_trace()
            hidden = (hactiv, cell)
            if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
                raise ValueError("Nan detected !")

        elif self.type== 'lstmplastic':
            fgt = F.sigmoid(self.x2f(input) + self.h2f(hidden[0]))
            ipt = F.sigmoid(self.x2i(input) + self.h2i(hidden[0]))
            opt = F.sigmoid(self.x2opt(input) + self.h2opt(hidden[0]))
            #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(input) + self.h2c(hidden[0])))
            
            #Need to think what the inputs and outputs should be for the
            #plasticity. It might be worth introducing an additional stage
            #consisting of whatever is multiplied by ift and then added to the
            #cell state, rather than the full cell state.... But we can
            #experiment both!
            
            #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(input) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))) #  self.h2c(hidden[0])))
            inputtocell =  F.tanh(self.x2c(input) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputtocell) #  self.h2c(hidden[0])))


            if self.rule == 'hebb':
                #hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden[0].unsqueeze(2), cell.unsqueeze(1))[0]
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden[0].unsqueeze(2), inputtocell.unsqueeze(1))[0]
            elif self.rule == 'oja':
                # NOTE: NOT SURE ABOUT THE OJA VERSION !!
                hebb = hebb + self.eta * torch.mul((hidden[0][0].unsqueeze(1) - torch.mul(hebb , inputtocell[0].unsqueeze(0))) , inputtocell[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                #hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
            hactiv = torch.mul(opt, F.tanh(cell))
            #pdb.set_trace()
            hidden = (hactiv, cell)
            if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
                raise ValueError("Nan detected !")

        elif self.type == 'plastic':
            hactiv = self.activ(self.i2h(input) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
            elif self.rule == 'oja':
                hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = hactiv
        
        elif self.type == 'homo':
            hactiv = self.activ(self.i2h(input) + hidden.mm(self.w + self.alpha * hebb))
            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
            elif self.rule == 'oja':
                hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = hactiv
        
        activout = self.softmax(self.h2o(hactiv))   # Action selection
        valueout = self.h2v(hactiv)                 # Value prediction (for A3C)

        return activout, valueout, hidden, hebb

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['hiddensize'], self.params['hiddensize']) , requires_grad=False).cuda()

    def initialZeroState(self):
        if self.params['type'] == 'lstm':
            return (Variable(torch.zeros(1, 1, self.params['hiddensize']), requires_grad=False).cuda() , Variable(torch.zeros(1, 1, self.params['hiddensize']), requires_grad=False ).cuda() )
        elif self.params['type'] == 'lstmmanual' or self.params['type'] == 'lstmplastic':
            return (Variable(torch.zeros(1, self.params['hiddensize']), requires_grad=False).cuda() , Variable(torch.zeros(1, self.params['hiddensize']), requires_grad=False ).cuda() )
        elif self.params['type'] == 'rnn' or self.params['type'] == 'plastic' or self.params['type'] == 'homo':
            return Variable(torch.zeros(1, self.params['hiddensize']), requires_grad=False ).cuda() 
        else:
            raise ValueError("Which type?")



def train(paramdict):
    #params = dict(click.get_current_context().params)
    print("Starting training...")
    params = {}
    #params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "maze_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])

    print("Initializing network")
    net = Network(params)
    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

    LABSIZE = params['labsize'] 
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
    lab[CTR,CTR] = 0 # Not really necessary, but nicer to not start on a wall, and perhaps helps localization by introducing a detectable irregularity in the center?



    all_losses = []
    all_losses_objective = []
    all_losses_eval = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    
    print("Starting episodes...")
    sys.stdout.flush()

    pos = 0
    hidden = net.initialZeroState()
    hebb = net.initialZeroHebb()


    # Starting episodes!
    
    for numiter in range(params['nbiter']):
        
        PRINTTRACE = 0
        if (numiter+1) % (1 + params['print_every']) == 0:
            PRINTTRACE = 1

        # Note: it doesn't matter if the reward is on the center (reward is only computed after an action is taken). All we need is not to put it on a wall or pillar (lab=1)
        rposr = 0; rposc = 0
        if params['rp'] == 0:   
            # If we want to constrain the reward to fall on the periphery of the maze
            while lab[rposr, rposc] == 1:
                rposr = np.random.randint(1, LABSIZE - 1)
                rposc = np.random.randint(1, LABSIZE - 1)
        elif params['rp'] == 1:
            while lab[rposr, rposc] == 1 or (rposr != 1 and rposr != LABSIZE -2 and rposc != 1 and rposc != LABSIZE-2):
                rposr = np.random.randint(1, LABSIZE - 1)
                rposc = np.random.randint(1, LABSIZE - 1)
        #print("Reward pos:", rposr, rposc)

        # Agent always starts an episode from the center
        posc = CTR
        posr = CTR

        optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()


        reward = 0.0
        rewards = []
        vs = []
        logprobs = []
        sumreward = 0.0
        dist = 0

        for numstep in range(params['eplen']):
            
            
            inputsN = np.zeros((1, TOTALNBINPUTS), dtype='float32')
            inputsN[0, 0:RFSIZE * RFSIZE] = lab[posr - RFSIZE//2:posr + RFSIZE//2 +1, posc - RFSIZE //2:posc + RFSIZE//2 +1].flatten()
            
            inputs = torch.from_numpy(inputsN).cuda()
            # Previous chosen action
            #inputs[0][numactionchosen] = 1
            inputs[0][-1] = 1 # Bias neuron
            inputs[0][-2] = numstep
            inputs[0][-3] = reward
            
            # Running the network
            y, v, hidden, hebb = net(Variable(inputs, requires_grad=False), hidden, hebb)  # y  should output probabilities; v is the value prediction 
        
            distrib = torch.distributions.Categorical(y)
            actionchosen = distrib.sample()  # sample() returns a Pytorch tensor of size 1; this is needed for the backprop below
            numactionchosen = actionchosen.data[0]    # Turn to scalar

            # Target position, based on the selected action
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
                reward = -.1
            else:
                dist += 1
                posc = tgtposc
                posr = tgtposr

            # Did we hit the reward location ? Increase reward and teleport!
            # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move
            if rposr == posr and rposc == posc:
                reward += 10
                if params['randstart'] == 1:
                    posr = np.random.randint(1, LABSIZE - 1)
                    posc = np.random.randint(1, LABSIZE - 1)
                    while lab[posr, posc] == 1:
                        posr = np.random.randint(1, LABSIZE - 1)
                        posc = np.random.randint(1, LABSIZE - 1)
                else:
                    posr = CTR
                    posc = CTR


            # Store the obtained reward, value prediction, and log-probabilities, for this time step
            rewards.append(reward)
            sumreward += reward
            vs.append(v)
            logprobs.append(distrib.log_prob(actionchosen))

            # A3C/A2C has an entropy reward on the output probabilities, to
            # encourage exploration. Our version of PyTorch does not have an
            # entropy() function for Distribution, so we use a penalty on the
            # sum of squares instead, which has the same basic property
            # (discourages concentration). It really does help!
            loss += params['bentropy'] * y.pow(2).sum()   

            #if PRINTTRACE:
            #    print("Probabilities:", y.data.cpu().numpy(), "Picked action:", numactionchosen, ", got reward", reward)


        # Do the A2C ! (essentially copied from V. Mnih, https://arxiv.org/abs/1602.01783, Algorithm S3)
        R = 0
        gammaR = params['gr']
        for numstepb in reversed(range(params['eplen'])) :
            R = gammaR * R + rewards[numstepb]
            lossv += (vs[numstepb][0] - R).pow(2) 
            loss -= logprobs[numstepb] * (R - vs[numstepb].data[0][0])  



        if PRINTTRACE:
            print("lossv: ", lossv.data.cpu().numpy()[0])
            print ("Total reward for this episode:", sumreward, "Dist:", dist)

        # Do we want to squash rewards for stabilization? 
        if params['squash'] == 1:
            if sumreward < 0:
                sumreward = -np.sqrt(-sumreward)
            else:
                sumreward = np.sqrt(sumreward)
        elif params['squash'] == 0:
            pass
        else:
            raise ValueError("Incorrect value for squash parameter")

        # Mixing the reward loss and the value-prediction loss
        loss += params['blossv'] * lossv
        loss /= params['eplen']
        loss.backward()

        #scheduler.step()
        optimizer.step()
        #torch.cuda.empty_cache()  

        lossnum = loss.data[0]
        lossbetweensaves += lossnum
        if (numiter + 1) % 10 == 0:
            all_losses_objective.append(lossnum)
            all_losses_eval.append(sumreward)
            all_losses_v.append(lossv.data[0])



        # Algorithm done. Now print statistics and save files.

        if (numiter+1) % params['print_every'] == 0:

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['print_every'])
            lossbetweensaves = 0
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)
            if params['type'] == 'plastic' or params['type'] == 'lstmplastic':
                print("ETA: ", net.eta.data.cpu().numpy(), "alpha[0,1]: ", net.alpha.data.cpu().numpy()[0,1], "w[0,1]: ", net.w.data.cpu().numpy()[0,1] )
            elif params['type'] == 'rnn':
                print("w[0,1]: ", net.w.data.cpu().numpy()[0,1] )

        if (numiter+1) % params['save_every'] == 0:
            print("Saving files...")
            losslast100 = np.mean(all_losses_objective[-100:])
            print("Average loss over the last 100 episodes:", losslast100)
            print("Saving local files...")
            with open('params_'+suffix+'.dat', 'wb') as fo:
                    pickle.dump(params, fo)
            with open('lossv_'+suffix+'.txt', 'w') as thefile:
                for item in all_losses_v:
                        thefile.write("%s\n" % item)
            with open('loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_losses_eval:
                        thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            # Uber-only
            print("Saving HDFS files...")
            if checkHdfs():
                print("Transfering to HDFS...")
                transferFileToHdfsDir('loss_'+suffix+'.txt', '/ailabs/tmiconi/gridlab/')
                transferFileToHdfsDir('torchmodel_'+suffix+'.dat', '/ailabs/tmiconi/gridlab/')
                transferFileToHdfsDir('params_'+suffix+'.dat', '/ailabs/tmiconi/gridlab/')
            #print("Saved!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    #parser.add_argument("--clamp", type=float, help="maximum (absolute value) gradient for clamping", default=1000000.0)
    parser.add_argument("--bentropy", type=float, help="coefficient for the A2C 'entropy' reward (really Simpson index concentration measure)", default=0.1)
    parser.add_argument("--blossv", type=float, help="coefficient for the A2C value prediction loss", default=.03)
    parser.add_argument("--labsize", type=int, help="size of the labyrinth; must be odd", default=9)
    parser.add_argument("--randstart", type=int, help="when hitting reward, should we teleport to random location (1) or center (0)?", default=1)
    parser.add_argument("--rp", type=int, help="whether the reward should be on the periphery", default=0)
    parser.add_argument("--squash", type=int, help="squash reward through signed sqrt (1 or 0)", default=0)
    #parser.add_argument("--nbarms", type=int, help="number of arms", default=2)
    #parser.add_argument("--nbseq", type=int, help="number of sequences between reinitializations of hidden/Hebbian state and position", default=3)
    parser.add_argument("--activ", help="activ function ('tanh' or 'selu')", default='tanh')
    parser.add_argument("--rule", help="learning rule ('hebb' or 'oja')", default='oja')
    parser.add_argument("--type", help="network type ('rnn' or 'plastic')", default='rnn')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=250)
    parser.add_argument("--hiddensize", type=int, help="size of the recurrent (hidden) layer", default=200)
    #parser.add_argument("--steplr", type=int, help="duration of each step in the learning rate annealing schedule", default=100000000)
    #parser.add_argument("--gamma", type=float, help="learning rate annealing factor", default=0.3)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=200)
    parser.add_argument("--print_every", type=int, help="number of cycles between successive printing of information", default=100)
    #parser.add_argument("--", type=int, help="", default=1e-4)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    #train()
    train(argdict)

