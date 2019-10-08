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





np.set_printoptions(precision=4)

NBDA = 1  # Number of different DA output neurons. At present, the code assumes NBDA=1 and will NOT WORK if you change this.


np.set_printoptions(precision=4)


ADDINPUT = 4 # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs

NBACTIONS = 4  # U, D, L, R

RFSIZE = 3 # Receptive Field

TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBACTIONS

##ttype = torch.FloatTensor;
#ttype = torch.cuda.FloatTensor;

##ttype = torch.FloatTensor;
#ttype = torch.cuda.FloatTensor;


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        self.rule = params['rule']
        self.type = params['type']
        self.softmax= torch.nn.functional.softmax
        #if params['activ'] == 'tanh':
        self.activ = F.tanh
        #elif params['activ'] == 'selu':
        #    self.activ = F.selu
        #else:
        #    raise ValueError('Must choose an activ function')
        if params['type'] == 'lstm':
            self.lstm = torch.nn.LSTM(TOTALNBINPUTS, params['hs']).cuda()
        elif params['type'] == 'rnn':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            #self.inputnegmask = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.inputnegmask[0, :TOTALNBINPUTS] = 0   # no modulation for 2nd half
        elif params['type'] == 'modplast' or params['type'] == 'modplast2':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            #self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True)
            #self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
            #self.inputnegmask = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.inputnegmask[0, :TOTALNBINPUTS] = 0   # no modulation for 2nd half
            self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
        elif params['type'] == 'plastic' :
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
            #self.inputnegmask = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.inputnegmask[0, :TOTALNBINPUTS] = 0   # no modulation for 2nd half
        elif params['type'] == 'modul' or params['type'] == 'modul2':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            # Note that initial eta is higher (faster) thanbefore
            self.eta = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
            self.etaet = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same etapw
            self.etapw = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same etapw
            self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
            # The daweights vectors are weight vectors from the DA output neurons to the network hidden (recurrent) neurons
            #self.daweights0 = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.daweights0[0, (params['hs'] // 2):] = 0   # no modulation for 2nd half
            #self.inputnegmask = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.inputnegmask[0, :TOTALNBINPUTS] = 0   # no modulation for 2nd half

            #else:
            #    raise ValueError("Must specify which half of the network receives modulation")
            self.daweights1 = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            self.daweights1[0, :(params['hs'] // 4)] = 0
            self.daweights1[0, -(params['hs'] // 4):] = 0
        else:
            raise ValueError("Which network type?")
        self.h2o = torch.nn.Linear(params['hs'], NBACTIONS).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()
        self.params = params

        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

    def forward(self, inputs, hidden, hebb, et, pw):
        BATCHSIZE = self.params['bs']
        HS = self.params['hs']

        if self.type == 'rnn':
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul(self.w.view(1, HS, HS),
                hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            hidden = hactiv
            #activout = self.softmax(self.h2o(hactiv))
            activout = self.h2o(hactiv)   # Linear!
            valueout = self.h2v(hactiv)
            #valueout = 0


        elif self.type == 'plastic':
            # Each row of w and hebb contains the input weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                            hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            if self.rule == 'hebb':
                deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
            elif self.rule == 'oja':
                deltahebb =  torch.mul(hactiv.view(BATCHSIZE, HS, 1), (hidden.view(BATCHSIZE, 1, HS) - torch.mul(self.w.view(1, HS, HS), hactiv.view(BATCHSIZE, HS, 1))))
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")

            if self.params['addpw'] == 3:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Hard clamp
                hebb = torch.clamp( hebb +  self.eta * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Soft clamp
                hebb = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            elif self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge. No decay/clamp.
                hebb = hebb + self.eta * deltahebb
            elif self.params['addpw'] == 0:
                # We do it the normal way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient.
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                if self.rule == 'hebb':
                    hebb = (1 - self.eta) * hebb + self.eta * deltahebb
                elif self.rule == 'oja':
                    hebb =  hebb + self.eta  * deltahebb

            hidden = hactiv


        elif self.type == 'modplast':
            # The actual network update should be the same as for "plastic". Only the Hebbian updates should be different
            # The columns of w and pw are the inputs weights to a single neuron
            hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w.view(HS, HS) + torch.mul(self.alpha.view(HS,HS), hebb)))
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            if self.rule == 'hebb':
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
            elif self.rule == 'oja':
                deltahebb = torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))

            if self.params['addpw'] == 3: # Hard clamp, purely additive
                # Note that we do the same for Hebb and Oja's rule
                hebb1 = torch.clamp(hebb + DAout[0,0] * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                hebb1 = torch.clamp( hebb +  torch.clamp(DAout[0,0] * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(DAout[0,0] * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            elif self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge
                # Note that we do the same for Hebb and Oja's rule
                hebb1 = hebb + DAout[0,0] * deltahebb

            elif self.params['addpw'] == 0:
                # We do it the normal way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient by grouping it with the computation of the other (non-modulated) half.
                # NOTE: This can go awry if DAout can go negative!
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                if self.rule == 'hebb':
                    hebb1 = (1 - DAout[0,0]) * hebb + DAout[0,0] * deltahebb
                elif self.rule == 'oja':
                    hebb1=  hebb + DAout[0,0] * deltahebb
            else:
                raise ValueError("Which additive form for plastic weights?")

            # The non-neuromodulated half of the network just does standard plasticity, using learned self.eta.
            if self.rule == 'hebb':
                hebb2 = (1 - self.eta) * hebb + self.eta * deltahebb
            elif self.rule == 'oja':
                hebb2 = hebb + self.eta * deltahebb
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")

            if self.params['fm'] == 1:
                hebb = hebb1
            elif self.params['fm'] == 0:
                hebb = torch.cat( (hebb1[:, :self.params['hs']//2], hebb2[:, self.params['hs']//2:]), dim=1)
            else:
                raise ValueError("Must select whether fully modulated or not")

            hidden = hactiv





        elif self.type == 'modplast_old':

            #Here we compute the same deltahebb for the whole network, and use
            #the same addpw for the whole network too.  #Only difference between
            #modulated and non-modulated halves is whether eta is the network's
            #(learned) eta parameter or the neuromodulator output DAout

            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                            hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            # deltahebb has shape BS x HS x HS
            # Each row of hebb contain the input weights to a neuron
            if self.rule == 'hebb':
                deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
            elif self.rule == 'oja':
                deltahebb =  torch.mul(hactiv.view(BATCHSIZE, HS, 1), (hidden.view(BATCHSIZE, 1, HS) - torch.mul(self.w.view(1, HS, HS), hactiv.view(BATCHSIZE, HS, 1))))
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")


            if self.params['addpw'] == 3: # Hard clamp, purely additive
                # Note that we do the same for Hebb and Oja's rule
                hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
                hebb2 = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                hebb1 = torch.clamp( hebb +  torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=0.0) * (1 - hebb) +
                        torch.clamp(DAout.view(BATCHSIZE, 1, 1)  * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
                hebb2 = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            elif self.params['addpw'] == 1: # Purely additive. This will almost certainly diverge, don't use it!
                hebb1 = hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
                hebb2 = hebb + self.eta * deltahebb

            elif self.params['addpw'] == 0:
                # We do it the old way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient
                # NOTE: THIS WILL GO AWRY if DAout is allowed to go outside [0,1]!
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                if self.rule == 'hebb':
                    hebb1 = (1 - DAout.view(BATCHSIZE,1,1)) * hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
                    hebb2 = (1 - self.eta) * hebb + self.eta *  deltahebb
                elif self.rule == 'oja':
                    hebb1=  hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
                    hebb2=  hebb + self.eta * deltahebb
            else:
                raise ValueError("Which additive form for plastic weights?")

            if self.params['fm'] == 1:
                hebb = hebb1
            elif self.params['fm'] == 0:
                hebb = torch.cat( (hebb1[:, :self.params['hs']//2, :], hebb2[:,  self.params['hs'] // 2:, :]), dim=1) # Maybe along dim=2 instead?...
            else:
                raise ValueError("Must select whether fully modulated or not")

            hidden = hactiv


        elif self.type == 'modul':

            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                            hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")


            # We need to select the order of operations; network update, e.t. update, neuromodulated incorporation into plastic weights
            # One possibility (for now go with this one):
            #    - computing all outputs from current inputs, including DA
            #    - incorporating neuromodulated Hebb/eligibility trace into plastic weights
            #    - computing updated hebb/eligibility traces
            # Another possibility (modul2):
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb/eligibility traces
            #    - incorporating this modified Hebb into plastic weights through neuromodulation


            # For Hebb (not et or pw); this is only used if fm=0, for the non-modulated part of the network
            # If fm=0:
            # One half of the network receives neuromodulation. The other just
            # does plain Hebbian plasticity; note that the eta's for the
            # Hebbian trace and the eligibility trace are different
            if self.params['fm']==0:
                if self.rule == 'hebb':
                    deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
                elif self.rule == 'oja':
                    deltahebb =  torch.mul(hactiv.view(BATCHSIZE, HS, 1), (hidden.view(BATCHSIZE, 1, HS) - torch.mul(self.w.view(1, HS, HS), hactiv.view(BATCHSIZE, HS, 1))))
                else:
                    raise ValueError("Must specify learning rule ('hebb' or 'oja')")

            # In modul2 we compute deltaet and update et here too; here we compute them later

            if self.params['addpw'] == 3:
                # Hard clamp
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
                if self.params['fm']==0:
                    hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                # This constrains the pw to stay within [-1, 1] (we could also do that by putting a tanh on top of it, but instead we want pw itself to remain within that range, to avoid large gradients and facilitate movement back to 0)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly" constrained to stay within [-1, 1], but finite-size effects might throw it off.
                pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw) +  torch.clamp(deltapw, max=0.0) * (pw + 1) , min=-.99999, max=.99999)
                if self.params['fm']==0:
                    hebb = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            if self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                pw1 = pw + deltapw
                if self.params['fm'] == 0:
                    hebb = hebb + self.eta * deltahebb
            elif self.params['addpw'] == 0:
                # We do it the old way, with a decay term.
                # This will FAIL if DAout is allowed to go outside [0,1]
                # Note: this makes the plastic weights decaying!
                pw1 = (1 - DAout.view(BATCHSIZE,1,1)) * pw1 + DAout.view(BATCHSIZE, 1, 1) * et
                if self.params['fm']==0:
                    if self.rule == 'hebb':
                        hebb = (1 - self.eta) * hebb + self.eta * deltahebb
                    elif self.rule == 'oja':
                        hebb=  hebb + self.eta * deltahebb
            # Should we have a fully neuromodulated network, or only half?
            if self.params['fm'] == 1:
                pw = pw1
            elif self.params['fm'] == 0:
                pw = torch.cat( (hebb[:, :self.params['hs']//2, :], pw1[:,  self.params['hs'] // 2:, :]), dim=1) # Maybe along dim=2 instead?...
            else:
                raise ValueError("Must select whether fully modulated or not")

            # Updating the eligibility trace - always a simple decay term.
            # Note that self.etaet != self.eta (which is used for hebb, i.e. the non-modulated part)
            deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
            et = (1 - self.etaet) * et + self.etaet *  deltaet

            hidden = hactiv





        elif self.type == 'modul2':

            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                            hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")


            # We need to select the order of operations; network update, e.t. update, neuromodulated incorporation into plastic weights
            # One possibility (for now go with this one):
            #    - computing all outputs from current inputs, including DA
            #    - incorporating neuromodulated Hebb/eligibility trace into plastic weights
            #    - computing updated hebb/eligibility traces
            # Another possibility (modul2):
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb/eligibility traces
            #    - incorporating this modified Hebb into plastic weights through neuromodulation


            # For Hebb (not et or pw); this is only used if fm=0, for the non-modulated part of the network
            # If fm=0:
            # One half of the network receives neuromodulation. The other just
            # does plain Hebbian plasticity; note that the eta's for the
            # Hebbian trace and the eligibility trace are different
            if self.params['fm']==0:
                if self.rule == 'hebb':
                    deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
                elif self.rule == 'oja':
                    deltahebb =  torch.mul(hactiv.view(BATCHSIZE, HS, 1), (hidden.view(BATCHSIZE, 1, HS) - torch.mul(self.w.view(1, HS, HS), hactiv.view(BATCHSIZE, HS, 1))))
                else:
                    raise ValueError("Must specify learning rule ('hebb' or 'oja')")

            # Updating the eligibility trace - always a simple decay term.
            # Note that self.etaet != self.eta (which is used for hebb, i.e. the non-modulated part)
            deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
            et = (1 - self.etaet) * et + self.etaet *  deltaet

            if self.params['addpw'] == 3:
                # Hard clamp
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
                if self.params['fm']==0:
                    hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                # This constrains the pw to stay within [-1, 1] (we could also do that by putting a tanh on top of it, but instead we want pw itself to remain within that range, to avoid large gradients and facilitate movement back to 0)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly" constrained to stay within [-1, 1], but finite-size effects might throw it off.
                pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw) +  torch.clamp(deltapw, max=0.0) * (pw + 1) , min=-.99999, max=.99999)
                if self.params['fm']==0:
                    hebb = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            if self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout.view(BATCHSIZE,1,1) * et
                pw1 = pw + deltapw
                if self.params['fm'] == 0:
                    hebb = hebb + self.eta * deltahebb
            elif self.params['addpw'] == 0:
                # We do it the old way, with a decay term.
                # This will FAIL if DAout is allowed to go outside [0,1]
                # Note: this makes the plastic weights decaying!
                pw1 = (1 - DAout.view(BATCHSIZE,1,1)) * pw1 + DAout.view(BATCHSIZE, 1, 1) * et
                if fm==0:
                    if self.rule == 'hebb':
                        hebb = (1 - self.eta) * hebb + self.eta * deltahebb
                    elif self.rule == 'oja':
                        hebb=  hebb + self.eta * deltahebb
            # Should we have a fully neuromodulated network, or only half?
            if self.params['fm'] == 1:
                pw = pw1
            elif self.params['fm'] == 0:
                pw = torch.cat( (hebb[:, :self.params['hs']//2, :], pw1[:,  self.params['hs'] // 2:, :]), dim=1) # Maybe along dim=2 instead?...
            else:
                raise ValueError("Must select whether fully modulated or not")


            hidden = hactiv





        return activout, valueout, hidden, hebb, et, pw



    def initialZeroHebb(self):
        #return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()
        return Variable(torch.zeros(self.params['hs'], self.params['hs']) , requires_grad=False).cuda()
    def initialZeroPlasticWeights(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()

    def initialZeroState(self):
        BATCHSIZE = self.params['bs']
        return Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False ).cuda()



def train(paramdict):
    #params = dict(click.get_current_context().params)

    #TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    #params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "btch_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)

    print("Initializing network")
    net = Network(params)
    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    #total_loss = 0.0
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=1e-4, weight_decay=params['l2'])
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


    print("Starting episodes!")

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
            # Note: it doesn't really matter if the reward is on the center. All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0; myrposc = 0
            # This one is for positioning the reward only in the periphery!
            #while lab[myrposr, myrposc] == 1 or (myrposr != 1 and myrposr != LABSIZE -2 and myrposc != 1 and myrposc != LABSIZE-2):
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr; rposc[nb] = myrposc
            #print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR

        optimizer.zero_grad()
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



            ## We randomly relocate the reward halfway through
            #if numstep == reloctime:
            #    rposr = 0; rposc = 0
            #    while lab[rposr, rposc] == 1 or (rposr == CTR and rposc == CTR):
            #        rposr = np.random.randint(1, LABSIZE - 1)
            #        rposc = np.random.randint(1, LABSIZE - 1)


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
            #if numiter == 115 and numstep == 99: identical
            #if numiter == 125 and numstep == 99: diff
            #if numiter == 120 and numstep == 99: identical
            #if numiter == 122 and numstep == 99:  diff
            #if numiter == 121 and numstep == 99: identical
            #if numiter == 122 and numstep == 14:  diff (a little, ~1e-3)
            #if numiter == 122 and numstep == 11:  diff (2e-2), rposr,rposc identical, posr different (5 vs 9 for batch)
            #if numiter == 122 and numstep == 10:  identical, rposr 5, rposc 6, posr 5, posc 6 for both
            ####
            #if numiter == 730 and numstep == 12: diff
            #if numiter == 700 and numstep == 12:  # diff (not by much.. in the y)
            #if numiter == 600 and numstep == 12:  # identical ... or so I thought?
            #if numiter == 650 and numstep == 12:  # diff (1e-6)
            #if numiter == 625 and numstep == 12:  # diff (1e-5)
            #if numiter == 612 and numstep == 12:  # diff (1e-6)
            #if numiter == 606 and numstep == 12:  # diff (1e-7)
            #if numiter == 603 and numstep == 12:  # diff (1e-6)
            #if numiter == 601 and numstep == 12:  # diff
            #if numiter == 600 and numstep == 99: # diff
            #if numiter == 600 and numstep == 15: #diff
            #if numiter == 600 and numstep == 1: #diff
            #if numiter == 500 and numstep == 1: diff
            #if numiter == 152 and numstep == 1: identical
            #if numiter == 352 and numstep == 1: # diff
            #if numiter == 252 and numstep == 1:  # identical!
            #if numiter == 302 and numstep == 1:  # identical!
            #if numiter == 332 and numstep == 1:   # diff
            #if numiter == 316 and numstep == 1:   # diff
            #if numiter == 309 and numstep == 1:  # diff
            #if numiter == 304 and numstep == 1:   # identical
            #if numiter == 306 and numstep == 1:   # identical
            #if numiter == 308 and numstep == 1:  # diff
            #if numiter == 307 and numstep == 1:  # diff
            #if numiter == 306 and numstep == 51:  # diff
            #if numiter == 306 and numstep == 21:  # diff
            #if numiter == 306 and numstep == 1:  # identical (confirm)
            #if numiter == 306 and numstep == 5:  #diff
            #if numiter == 306 and numstep == 3:  # diff
            #if numiter == 306 and numstep == 2:  # identical, rposc rposr  3,4, posc posr 5, 3... hebb noticeably diff! 1e-6; alpha/w identical, h2o(hidden) identical
            #if numiter == 306 and numstep == 1:  h2da(hidden) identical, but h2v(hidden) different! h2o(hidden) identical, hebb different... h2v has identical weights+biases though! hidden identical...
            # wait, hidden NOT identical - pow(2).sum gives exact same result to 36 decimals, but hidden[0,25] does not!
            #if numiter == 305 and numstep == 1:  # lol, hidden[0,2] is different....
            #if numiter == 150 and numstep == 99:  # hidden[0,2] different, event though abss().sum(): identical
            #if numiter == 99 and numstep == 99:   # hidden[0,2] identical...
            #if numiter == 101 and numstep == 99:   # various components of hidden identical
            #if numiter == 221 and numstep == 99:   # hidden different; the difference seems to be caused by loss.backward/optimizer.step.. and disappears if lossv is commented out?!  blossv=0 also removes it! vs[15][0] also different (with blossv=0, so no diff in hidden, and no diff in h2v either!) vs[0][0] identical, vs[1][0] different.....(by ~1e-8) again with blossv=0... if I try with normal blossv but preventing loss.backward/optimizer.step, then I get identical vs[2][0]/vs[-1][0]... if I comment out blossv*lossv addition: hidden identical, vs[-1][0] different, h2DA identical, h2v dot hidden.t() identical... v is identical but the vs are different, how can that be?? if I put the set_trace just after vs.append(v), v[-1][0] is identical, but v[2][0] is not.. and neither is vs[-2][0] !!
            #if numiter == 221 and numstep == 98, interup just after vs.append(v), blossv=0: now vs[-1][0] also different, as is v... hidden is different too! Confirmed that if you stop at 99 they are identical (How!!!)

            #if numiter == 121 and numstep == 98: # identical
            #if numiter == 151 and numstep == 98: # identical
            #if numiter == 191 and numstep == 98: #identical
            #if numiter == 208 and numstep == 98:
            #if numiter == 215 and numstep == 98:
            #if numiter == 218 and numstep == 98: # all identical
            #if numiter == 220 and numstep == 98: #h identical, but vs[-]1[0] different!
            #if numiter == 218 and numstep == 98: #h identical, including vs[-1][0]... but not vs[-21][0] ! vs[2][0] identical though! Lol, all vs identical except vs[-21][0]...
            #if numiter == 218 and numstep == 77: #h identical,but v different (vs[-1][0] identical. as expect, net.h2v(hidden) different, h2v.weight dot hidden different... but h2v weight/bias have identical abs sum, and so does hidden! torch.matmul(hidden[0,0:14] , net.h2v.weight[0,0:14]) identical, but :15 different!  torch.sum(hidden[0,0:15] - net.h2v.weight[0,0:15]) identical... but if you replace - with * or +, different! hidden[0,14] is different! lol, hidden.sum() and hidden.abs().sum() are identical,  hidden[0,0:].sum()/abs().sum() identical, but hidden[0,0:24].sum() is different! hidden[0,24:].sum() is identical too... BASICALLY, the w's and alphas have several differences in the 1e-9 range; the h2v don't

            #if numiter == 120 and numstep == 98: # w's are already different...
            #if numiter == 101 and numstep == 98: #w's identical
            #if numiter == 102 and numstep == 98: #w's identical
            #if numiter == 103 and numstep == 98: # # w's differ in the 1e-10 range 
            #    pdb.set_trace()

            # torch.set_printoptions(precision=30)
            # np.savetxt('a2.txt', all_losses_objective)
            # p "{:.36f}".format(hidden.abs().sum().data.cpu().numpy()[0])  # Can also give identical results despite some different components??
            # BAD - may erase too small differences in individual components (bc of squaring) p "{:.36f}".format(net.h2DA(hidden).pow(2).sum().data.cpu().numpy()[0])
            # p "{:.36f}".format(hidden[0,2].data.cpu().numpy()[0])
            # p "{:.36f}".format(vs[-1][0].data.cpu().numpy()[0])

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

            #loss += ( params['bent'] * y.pow(2).sum() / BATCHSIZE )  # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution. Note: .2 may be too strong, .04 may be too weak.
            loss +=  params['bent'] * y.pow(2).sum()  # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution. Note: .2 may be too strong, .04 may be too weak.
            #lossentmean  = .99 * lossentmean + .01 * ( params['bent'] * y.pow(2).sum() / BATCHSIZE ).data[0] # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution. Note: .2 may be too strong, .04 may be too weak.


            if PRINTTRACE:
                #print("Step ", numstep, "- GI: ", goodinputs, ", GA: ", goodaction, " Inputs: ", inputsN, " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
                #        " - inputsthisstep:", inputsthisstep, " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Rew: ", reward)
                print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
                        " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Reward (this step, 1st in batch): ", reward[0])



        # Episode is done, now let's do the actual computations


        #R = Variable(torch.zeros(BATCHSIZE).cuda(), requires_grad=False)
        R = 0
        gammaR = params['gr']
        for numstepb in reversed(range(params['eplen'])) :
            #R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]).cuda(), requires_grad=False)
            #R = gammaR * R + float(rewards[numstepb][0])
            #ctrR = R - vs[numstepb][0]
            #lossv += ctrR.pow(2).sum() / BATCHSIZE
            #loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  # Need to check if detach() is OK
            R = gammaR * R + float(rewards[numstepb][0])
            lossv += (vs[numstepb][0] - R).pow(2)
            loss -= logprobs[numstepb] * (R - vs[numstepb].data[0][0])  # Not sure if the "data" is needed... put it b/c of worry about weird gradient flows


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

        if PRINTTRACE:
            if True: #params['algo'] == 'A3C':
                print("lossv: ", lossv.data.cpu().numpy()[0])
            print ("Total reward for this episode (all):", sumreward, "Dist:", dist)

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
            
            
        #if numiter == 102 :   # w identical, final hidden identical... but loss slightly different! as is lossv (shouldnt matter since its addition to loss is commented out). vs apparently identical
        #if numiter == 182 :   # identical loss (after fixing the rewards computations)
        #if numiter == 202 :   # identical loss (but loss_between_saves different, which means some losses different?...)
        #if numiter == 222 :    # loss is different
        #if numiter == 33 :    
        if numiter == 101 :    
            pdb.set_trace()

        #for p in net.parameters():
        #    p.grad.data.clamp_(-params['clp'], params['clp'])
        if numiter > 100:  # Burn-in period for meanrewards
            loss.backward()
            optimizer.step()

        #torch.cuda.empty_cache()

        #print(sumreward)
        lossnum = loss.data[0]
        lossbetweensaves += lossnum

        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())
            #all_losses_v.append(lossv.data[0])
        #total_loss  += lossnum


        if (numiter+1) % params['pe'] == 0:

            np.savetxt('a1.txt', all_losses_objective)

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ", np.sum(all_total_rewards[-params['pe']:])/ params['pe'])
            #print("Mean reward (across batch): ", sumreward.mean())
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            if params['type'] == 'plastic' or params['type'] == 'lstmplastic':
                print("ETA: ", net.eta.data.cpu().numpy(), "alpha[0,1]: ", net.alpha.data.cpu().numpy()[0,1], "w[0,1]: ", net.w.data.cpu().numpy()[0,1] )
            elif params['type'] == 'modul':
                print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy(), " mean-abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())))
            elif params['type'] == 'rnn':
                print("w[0,1]: ", net.w.data.cpu().numpy()[0,1] )

        if (numiter+1) % params['save_every'] == 0:
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
            with open('loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                        thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            with open('params_'+suffix+'.dat', 'wb') as fo:
                pickle.dump(params, fo)
            if os.path.isdir('/mnt/share/tmiconi'):
                print("Transferring to NFS storage...")
                for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
                    result = os.system(
                        'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/'+fn))
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
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.05)
    #parser.add_argument("--pen", type=float, help="penalty value (reward decrement for taking any non-rest action)", default=.2)
    #parser.add_argument("--exprew", type=float, help="reward value (reward increment for hitting reward location)", default=.0)
    parser.add_argument("--bent", type=float, help="coefficient for the entropy reward (really Simpson index concentration measure)", default=0.03)
    #parser.add_argument("--probarev", type=float, help="probability of reversal (random change) in desired stimulus-response, per time step", default=0.0)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    #parser.add_argument("--lsize", type=int, help="size of the labyrinth; must be odd", default=7)
    #parser.add_argument("--rp", type=int, help="whether the reward should be on the periphery", default=0)
    #parser.add_argument("--squash", type=int, help="squash reward through signed sqrt (1 or 0)", default=0)
    #parser.add_argument("--nbarms", type=int, help="number of arms", default=2)
    #parser.add_argument("--nbseq", type=int, help="number of sequences between reinitializations of hidden/Hebbian state and position", default=3)
    #parser.add_argument("--activ", help="activ function ('tanh' or 'selu')", default='tanh')
    #parser.add_argument("--algo", help="meta-learning algorithm (A3C or REI)", default='A3C')
    parser.add_argument("--rule", help="learning rule ('hebb' or 'oja')", default='hebb')
    parser.add_argument("--type", help="network type ('lstm' or 'rnn' or 'plastic')", default='modul')
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=9)
    parser.add_argument("--da", help="transformation function of DA signal (tanh or sig or lin)", default='tanh')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--fm", type=int, help="if using neuromodulation, do we modulate the whole network (1) or just half (0) ?", default=1)
    #parser.add_argument("--nu", type=float, help="REINFORCE baseline time constant", default=.1)
    #parser.add_argument("--samestep", type=int, help="compare stimulus and response in the same step (1) or from successive steps (0) ?", default=0)
    #parser.add_argument("--nbin", type=int, help="number of possible inputs stimulis", default=4)
    #parser.add_argument("--modhalf", type=int, help="which half of the recurrent netowkr receives modulation (1 or 2)", default=1)
    #parser.add_argument("--nbac", type=int, help="number of possible non-rest actions", default=4)
    parser.add_argument("--rsp", type=int, help="does the agent start each episode from random position (1) or center (0) ?", default=1)
    parser.add_argument("--addpw", type=int, help="are plastic weights purely additive (1) or forgetting (0) ?", default=1)
    #parser.add_argument("--clp", type=int, help="inputs clamped (1), fully clamped (2) or through linear layer (0) ?", default=0)
    #parser.add_argument("--md", type=int, help="maximum delay for reward reception", default=0)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    #parser.add_argument("--exptime", type=int, help="exploration (no reward) time (must be < eplen)", default=0)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=1)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=3e-6)
    #parser.add_argument("--steplr", type=int, help="duration of each step in the learning rate annealing schedule", default=100000000)
    #parser.add_argument("--gamma", type=float, help="learning rate annealing factor", default=0.3)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=1000)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information", default=100)
    #parser.add_argument("--", type=int, help="", default=1e-4)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    #train()
    train(argdict)

