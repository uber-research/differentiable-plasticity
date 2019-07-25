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

NBDA = 1


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
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
            #self.inputnegmask = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
            #self.inputnegmask[0, :TOTALNBINPUTS] = 0   # no modulation for 2nd half
            self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
            self.DAoutV = Variable(torch.ones(1, params['hs']), requires_grad=False).cuda()
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
        elif params['type'] == 'lstmplastic':
            self.h2f = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.h2i = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.h2opt = torch.nn.Linear(params['hs'], params['hs']).cuda()

            # Plasticity in the recurrent connections, h to c:
            #self.h2c = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.w =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).cuda(), requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta

            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
        elif params['type'] == 'lstmmanual':
            self.h2f = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.h2i = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.h2opt = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.h2c = torch.nn.Linear(params['hs'], params['hs']).cuda()
            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hs']).cuda()
            ##fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
            ##ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
            ##opt = F.sigmoid(self.x2o(inputs) + self.h2o(hidden[0]))
            ##cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))
            ##h = torch.mul(opt, cell)
            ##hidden = (h, cell)
        else:
            raise ValueError("Which network type?")
        self.h2o = torch.nn.Linear(params['hs'], NBACTIONS).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()
        self.params = params

        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

    def forward(self, inputs, hidden, hebb, et, pw):
        if self.type == 'lstm':
            hactiv, hidden = self.lstm(inputs.view(1, 1, -1), hidden)  # hactiv is just the h. hidden is the h and the cell state, in a tuple
            hactiv = hactiv[0]
            activout = self.softmax(self.h2o(hactiv))
            valueout = self.h2v(hactiv)
            #pdb.set_trace()
            #hactiv = hactiv.view(1, -1)  # Apparently this was causing memory leaks?.....

        # Draft for a "manual" lstm:
        elif self.type== 'lstmmanual':
            # hidden[0] is the previous h state. hidden[1] is the previous c state
            fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
            ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
            opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))
            hactiv = torch.mul(opt, F.tanh(cell))
            #pdb.set_trace()
            hidden = (hactiv, cell)
            activout = self.softmax(self.h2o(hactiv))
            valueout = self.h2v(hactiv)
            if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
                raise ValueError("Nan detected !")

        elif self.type== 'lstmplastic':
            fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
            ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
            opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
            #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))

            #Need to think what the inputs and outputs should be for the
            #plasticity. It might be worth introducing an additional stage
            #consisting of whatever is multiplied by ift and then added to the
            #cell state, rather than the full cell state.... But we can
            #experiment both!
            #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))) #  self.h2c(hidden[0])))
            inputstocell =  F.tanh(self.x2c(inputs) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputstocell) #  self.h2c(hidden[0])))

            if self.rule == 'hebb':
                raise ValueError("Not yet implemented!")
                #hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden[0].unsqueeze(2), cell.unsqueeze(1))[0]
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden[0].unsqueeze(2), inputstocell.unsqueeze(1))[0]
            elif self.rule == 'oja':
                raise ValueError("Not yet implemented!")
                # NOTE: NOT SURE ABOUT THE OJA VERSION !!
                hebb = hebb + self.eta * torch.mul((hidden[0][0].unsqueeze(1) - torch.mul(hebb , inputstocell[0].unsqueeze(0))) , inputstocell[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                #hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
            hactiv = torch.mul(opt, F.tanh(cell))
            #pdb.set_trace()
            hidden = (hactiv, cell)
            if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
                raise ValueError("Nan detected !")
            activout = self.softmax(self.h2o(hactiv))
            valueout = self.h2v(hactiv)





        elif self.type == 'rnn':
            if self.params['clp'] == 0:
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w))
            elif self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w))
            #elif self.params['clp'] == 2:
            #    hidden = self.inputnegmask.mul(hidden) + inputs
            #    hactiv = self.activ(hidden.mm(self.w))
            #    hactiv = self.inputnegmask.mul(hactiv) + inputs
            hidden = hactiv
            #activout = self.softmax(self.h2o(hactiv))
            activout = self.h2o(hactiv)   # Linear!
            valueout = self.h2v(hactiv)
            #valueout = 0

        elif self.type == 'plastic_prev':
            # The columns of w and pw are the inputs weights to a single neuron
            if self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            elif self.params['clp'] == 0:  # No clamping, input layer
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
            elif self.rule == 'oja':
                hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = hactiv

        elif self.type == 'plastic':
            # The columns of w and pw are the inputs weights to a single neuron (?)
            if self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            elif self.params['clp'] == 0:  # No clamping, input layer
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            if self.rule == 'hebb':
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
            elif self.rule == 'oja':
                deltahebb = torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))

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

        elif self.type == 'modplast2':

            #Here we compute the same deltahebb for the whole network, and use
            #the same addpw for the whole network too.  #Only difference between
            #modulated and non-modulated halves is whether eta is the network's
            #(learned) eta parameter or the neuromodulator output DAout

            # The columns of w and pw are the inputs weights to a single neuron
            if self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            else:  # No clamping, input layer
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
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
                hebb2 = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                hebb1 = torch.clamp( hebb +  torch.clamp(DAout[0,0] * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(DAout[0,0] * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
                hebb2 = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
            elif self.params['addpw'] == 1: # Purely additive. This will almost certainly diverge, don't use it!
                hebb1 = hebb + DAout[0,0] * deltahebb
                hebb2 = hebb + self.eta * deltahebb

            elif self.params['addpw'] == 0:
                # We do it the normal way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient
                # Note: This can go awry if DAout can go negative or outside [0,1]!
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                if self.rule == 'hebb':
                    hebb1 = (1 - DAout[0,0]) * hebb + DAout[0,0] * deltahebb
                    hebb2 = (1 - self.eta) * hebb + DAout[0,0] * deltahebb
                elif self.rule == 'oja':
                    hebb1=  hebb + DAout[0,0] * deltahebb
                    hebb2=  hebb + self.eta * deltahebb
            else:
                raise ValueError("Which additive form for plastic weights?")

            if self.params['fm'] == 1:
                hebb = hebb1
            elif self.params['fm'] == 0:
                hebb = torch.cat( (hebb1[:, :self.params['hs']//2], hebb2[:, self.params['hs']//2:]), dim=1)
            else:
                raise ValueError("Must select whether fully modulated or not")

            hidden = hactiv


        elif self.type == 'modplast':
            # The actual network update should be the same as for "plastic". Only the Hebbian updates should be different
            # The columns of w and pw are the inputs weights to a single neuron
            if self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            else:  # No clamping, input layer
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
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


        elif self.type == 'modul':

            # One half of the network receives neuromodulation. The other just
            # does plain Hebbian plasticity; note that the eta's for the
            # Hebbian trace and the eligibility trace are different

            # We need to select the order of operations; network update, hebb update, neuromodulated incorporation into stable plastic weights
            # One possibility (for now go with this one):
            #    - computing all outputs from current inputs, including DA
            #    - incorporating neuromodulated Hebb/eligibility trace into plastic weights
            #    - computing updated hebb
            # Another possibility:
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb
            #    - incorporating this modified Hebb into plastic weights through neuromodulation

            # The columns of w and pw are the inputs weights to a single neuron
            if self.params['clp'] == 0:
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, pw)))
            elif self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, pw)))
            #else:
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)
            #valueout = 0
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            if self.params['addpw'] == 3:
                # Hard clamp
                deltapw = DAout[0,0] * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
            #if self.params['addpw'] == 3:
            #    # Constrained AND cubed: This makes the soft bounds "softer", so the values can come closer to -1 and 1.
            #    # Absolutely no difference in performance from addpw=2 !
            #    deltapw = DAout[0,0] * et
            #    pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw ** 3) +  torch.clamp(deltapw, max=0.0) * (pw ** 3 + 1) , min=-1.0, max=1.0)
            #    #if np.random.rand() < .05:
            #    #    pdb.set_trace()
            elif self.params['addpw'] == 2:
                deltapw = DAout[0,0] * et
                # This constrains the pw to stay within [-1, 1] (we could do that by putting a tanh on top of it, but we want pw itself to remain within that range to avoid large gradients)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly" constrained to stay within [-1, 1], but finite-size effects might throw it off.
                # Note that cubing pw in the boundary terms below would make the bounds "softer" and allow a wider range, but in practice it makes no difference in performance.
                pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw) +  torch.clamp(deltapw, max=0.0) * (pw + 1) , min=-.99999, max=.99999)
                #if np.random.rand() < .05:
                #    pdb.set_trace()
            if self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout[0,0] * et
                pw1 = pw + deltapw
            elif self.params['addpw'] == 0:
                # Problem: this makes the plastic weights decaying!
                pw1 = pw - torch.abs(self.etapw) * pw + self.etapw * DAout[0,0] * et

            # Should we have a fully neuromodulated network, or only half?
            if self.params['fm'] == 1:
                pw = pw1
            elif self.params['fm'] == 0:
                pw = torch.cat( (hebb[:, :self.params['hs']//2], pw1[:, self.params['hs'] // 2:]), dim=1) # Use output argument?
            else:
                raise ValueError("Must select whether fully modulated or not")

            # Note that the 'hebb' variable is only for the non-modulated part,
            # which is only used if params['fm'] == 0; also, hebb can be
            # updated Oja or decaying, but et is always decaying.
            if self.rule == 'hebb':
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
                hebb = (1 - self.eta) * hebb + self.eta * deltahebb
                et = (1 - self.etaet) * et + self.etaet *  deltahebb
            elif self.rule == 'oja':
                #raise ValueError("Not yet implemented!")
                hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
                et = (1 - self.etaet) * et + self.etaet *  deltahebb
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = hactiv
            #if np.random.rand() < .05:
            #   pdb.set_trace()


        elif self.type == 'modul2':

            # Here we try the other order:
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb
            #    - incorporating this modified Hebb into plastic weights through neuromodulation

            # The columns of w and pw are the inputs weights to a single neuron
            if self.params['clp'] == 0:
                hactiv = self.activ(self.i2h(inputs) + hidden.mm(self.w + torch.mul(self.alpha, pw)))
            elif self.params['clp'] == 1:
                hactiv = self.activ(inputs + hidden.mm(self.w + torch.mul(self.alpha, pw)))
            #else:
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)
            #valueout = 0
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))
            elif self.params['da'] == 'lin':
                DAout =  self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            # Updating ET before PW (note: 'hebb' variable is only for the non-modulated part of the network)
            if self.rule == 'hebb':
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
                hebb = (1 - self.eta) * hebb + self.eta * deltahebb
                et = (1 - self.etaet) * et + self.etaet *  deltahebb
            elif self.rule == 'oja':
                #raise ValueError("Not yet implemented!")
                hebb = hebb + self.eta * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , hactiv[0].unsqueeze(0))) , hactiv[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                deltahebb = torch.bmm(hidden.unsqueeze(2), hactiv.unsqueeze(1))[0]
                et = (1 - self.etaet) * et + self.etaet *  deltahebb
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")


            if self.params['addpw'] == 3:
                # Hard clamp
                deltapw = DAout[0,0] * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
            #if self.params['addpw'] == 3:
            #    # Constrained AND cubed: This makes the soft bounds "softer", so the values can come closer to -1 and 1.
            #    # Absolutely no difference in performance from addpw=2 !
            #    deltapw = DAout[0,0] * et
            #    pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw ** 3) +  torch.clamp(deltapw, max=0.0) * (pw ** 3 + 1) , min=-1.0, max=1.0)
            #    #if np.random.rand() < .05:
            #    #    pdb.set_trace()
            elif self.params['addpw'] == 2:
                deltapw = DAout[0,0] * et
                # This constrains the pw to stay within [-1, 1] (we could do that by putting a tanh on top of it, but we want pw itself to remain within that range to avoid large gradients)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly" constrained to stay within [-1, 1], but finite-size effects might throw it off.
                # Note that cubing pw in the boundary terms below would make the bounds "softer" and allow a wider range, but in practice it makes no difference in performance.
                pw1 = torch.clamp( pw +  torch.clamp(deltapw, min=0.0) * (1 - pw) +  torch.clamp(deltapw, max=0.0) * (pw + 1) , min=-.99999, max=.99999)
                #if np.random.rand() < .05:
                #    pdb.set_trace()
            if self.params['addpw'] == 1: # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout[0,0] * et
                pw1 = pw + deltapw
            elif self.params['addpw'] == 0:
                # Problem: this makes the plastic weights decaying!
                pw1 = pw - torch.abs(self.etapw) * pw + self.etapw * DAout[0,0] * et

            # Should we have a fully neuromodulated network, or only half?
            if self.params['fm'] == 1:
                pw = pw1
            elif self.params['fm'] == 0:
                pw = torch.cat( (hebb[:, :self.params['hs']//2], pw1[:, self.params['hs'] // 2:]), dim=1) # Use output argument?
            else:
                raise ValueError("Must select whether fully modulated or not")

            hidden = hactiv
            #if np.random.rand() < .05:
            #   pdb.set_trace()


        return activout, valueout, hidden, hebb, et, pw



    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['hs'], self.params['hs']) , requires_grad=False).cuda()
    def initialZeroPlasticWeights(self):
        return Variable(torch.zeros(self.params['hs'], self.params['hs']) , requires_grad=False).cuda()

    def initialZeroState(self):
        if self.params['type'] == 'lstm':
            return (Variable(torch.zeros(1, 1, self.params['hs']), requires_grad=False).cuda() , Variable(torch.zeros(1, 1, self.params['hs']), requires_grad=False ).cuda() )
        elif self.params['type'] == 'lstmmanual' or self.params['type'] == 'lstmplastic':
            return (Variable(torch.zeros(1, self.params['hs']), requires_grad=False).cuda() , Variable(torch.zeros(1, self.params['hs']), requires_grad=False ).cuda() )
        elif self.params['type'] == 'rnn' or self.params['type'] == 'plastic'  or self.params['type'] == 'modul' or self.params['type'] == 'modul2' or self.params['type'] == 'modplast' or self.params['type'] == 'modplast2':
            return Variable(torch.zeros(1, self.params['hs']), requires_grad=False ).cuda()



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
    suffix = "maz_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'print_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames

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
        #if (numiter+1) % (1 + params['print_every']) == 0:
        if (numiter+1) % (params['print_every']) == 0:
            PRINTTRACE = 1

        #lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        #count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        rposr = 0; rposc = 0
        while lab[rposr, rposc] == 1 or (rposr == CTR and rposc == CTR):
            rposr = np.random.randint(1, LABSIZE - 1)
            rposc = np.random.randint(1, LABSIZE - 1)

        # We always start the episode from the center (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        posc = CTR
        posr = CTR

        optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()
        et = net.initialZeroHebb() # Eligibility Trace is identical to Hebbian Trace in shape
        pw = net.initialZeroPlasticWeights()
        numactionchosen = 0


        reward = 0.0
        rewards = []
        vs = []
        logprobs = []
        sumreward = 0.0
        dist = 0
        rewarddelay = -1
        rewardpercep = 0

        #reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

        #print("EPISODE ", numiter)
        for numstep in range(params['eplen']):



            ## We randomly relocate the reward halfway through
            #if numstep == reloctime:
            #    rposr = 0; rposc = 0
            #    while lab[rposr, rposc] == 1 or (rposr == CTR and rposc == CTR):
            #        rposr = np.random.randint(1, LABSIZE - 1)
            #        rposc = np.random.randint(1, LABSIZE - 1)


            if params['clp'] == 0:
                inputs = np.zeros((1, TOTALNBINPUTS), dtype='float32')
            else:
                inputs = np.zeros((1, params['hs']), dtype='float32')

            labg = lab.copy()
            #labg[rposr, rposc] = -1  # The agent can see the reward if it falls within its RF
            inputs[0, 0:RFSIZE * RFSIZE] = labg[posr - RFSIZE//2:posr + RFSIZE//2 +1, posc - RFSIZE //2:posc + RFSIZE//2 +1].flatten() * 1.0

            # Previous chosen action
            inputs[0, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
            inputs[0, RFSIZE * RFSIZE +2] = numstep / params['eplen']
            #inputs[0, RFSIZE * RFSIZE +3] = 1.0 * reward # Reward from previous time step
            inputs[0, RFSIZE * RFSIZE +3] = 1.0 * rewardpercep
            inputs[0, RFSIZE * RFSIZE + ADDINPUT + numactionchosen] = 1
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
            actionchosen = distrib.sample()  # sample() returns a Pytorch tensor of size 1; this is needed for the backprop below
            numactionchosen = actionchosen.data[0]    # Turn to scalar


            #if numiter == 103 and numstep == 98: 
            #    pdb.set_trace()


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

            # Did we hit the reward location ?
            if rposr == posr and rposc == posc:
                reward += params['rew']
                if params['rsp'] == 1:
                    posr = np.random.randint(1, LABSIZE - 1)
                    posc = np.random.randint(1, LABSIZE - 1)
                    while lab[posr, posc] == 1 or (rposr == posr and rposc == posc):
                        posr = np.random.randint(1, LABSIZE - 1)
                        posc = np.random.randint(1, LABSIZE - 1)
                else:
                    posr = CTR
                    posc = CTR
            rewardpercep = reward
            # This is with reward delay. Not necessarily buggy, but it causes some divergences w/ batch due to the reward counter for not detecting rewards
            #    if rewarddelay < 0:  # Make sure that the reward delay counter is not active. NOTE: this can cause weirdnesses if e.g. re-teleporting multiple times on the reward location....?
            #        # If we already have hit the reward location, but haven't
            #        # perceived it / been transported yet, we don't care if we
            #        # do it again before the perception (and transportation)
            #        # has occurred
            #        reward += params['rew']  # That is the reward that meta-learning cares about - not the one perceived by the agent, which is delayed
            #        rewarddelay = 1 + np.random.randint(1 + params['md'])

            #rewardpercep = 0
            #if rewarddelay > -1:
            #    rewarddelay -= 1
            #if rewarddelay == 0:
            #    # Now we can perceive the reward (and teleport)!
            #    # NOTE: in this implementation, the agent only perceives the positive
            #    # rewards - not the 'pain' of hitting the walls. That's OK (not
            #    # something you *need* to learn within-life, outer loop can
            #    # learn it)!
            #    rewardpercep = params['rew']
            #    if params['rsp'] == 1:
            #        posr = np.random.randint(1, LABSIZE - 1)
            #        posc = np.random.randint(1, LABSIZE - 1)
            #        while lab[posr, posc] == 1 or (rposr == posr and rposc == posc):
            #            posr = np.random.randint(1, LABSIZE - 1)
            #            posc = np.random.randint(1, LABSIZE - 1)
            #    else:
            #        posr = CTR
            #        posc = CTR

            ## Explortion reward (actually a penalty on the normalized visit count of the new location)
            #count[posr, posc] += 1
            #reward -= (count[posr, posc] / np.sum(count)) * params['exprew']


            if PRINTTRACE:
                #print("Step ", numstep, "- GI: ", goodinputs, ", GA: ", goodaction, " Inputs: ", inputsN, " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
                #        " - inputsthisstep:", inputsthisstep, " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Rew: ", reward)
                print("Step ", numstep, " Inputs: ", inputs[0,:TOTALNBINPUTS], " - Outputs: ", y.data.cpu().numpy(), " - action chosen: ", numactionchosen,
                        " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Reward (this step): ", reward)
            rewards.append(reward)
            vs.append(v)
            sumreward += reward



            logprobs.append(distrib.log_prob(actionchosen))

            #if params['algo'] == 'A3C':
            loss += params['bent'] * y.pow(2).sum()   # We want to penalize concentration, i.e. encourage diversity; our version of PyTorch does not have an entropy() function for Distribution, so we use this instead.

            ##if PRINTTRACE:
            ##    print("Probabilities:", y.data.cpu().numpy(), "Picked action:", numactionchosen, ", got reward", reward)


        # Episode is done, now let's do the actual computations
        gammaR = params['gr']
        if True: #params['algo'] == 'A3C':
            R = 0
            for numstepb in reversed(range(params['eplen'])) :
                #BATCHSIZE = 1
                #R = gammaR * R + rewards[numstepb]
                #ctrR = R - vs[numstepb][0]
                #lossv += ctrR.pow(2).sum() / BATCHSIZE
                #loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  # Need to check if detach() is OK
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

        #meanrewards[rposr, rposc] = (1.0 - params['nu']) * meanrewards[rposr, rposc] + params['nu'] * sumreward
        #R = 0
        #for numstepb in reversed(range(params['eplen'])) :
        #    R = gammaR * R + rewards[numstepb]
        #    meanrewardstmp[rposr, rposc, numstepb] = (1.0 - params['nu']) * meanrewardstmp[rposr, rposc, numstepb] + params['nu'] * R

        loss /= params['eplen']

        if PRINTTRACE:
            if True: #params['algo'] == 'A3C':
                print("lossv: ", lossv.data.cpu().numpy()[0])
            print ("Total reward for this episode:", sumreward, "Dist:", dist)

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
        if numiter == 5212 : 
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
        all_total_rewards.append(sumreward)
            #all_losses_v.append(lossv.data[0])
        #total_loss  += lossnum


        if (numiter+1) % params['print_every'] == 0:
            
            
            np.savetxt('a2.txt', all_losses_objective)

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['print_every'])
            lossbetweensaves = 0
            print("Mean reward: ", np.sum(all_total_rewards[-params['print_every']:])/ params['print_every'])
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)
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
    #parser.add_argument("--randstart", type=int, help="when hitting reward, should we teleport to random location (1) or center (0)?", default=0)
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
    parser.add_argument("--clp", type=int, help="inputs clamped (1), fully clamped (2) or through linear layer (0) ?", default=0)
    parser.add_argument("--md", type=int, help="maximum delay for reward reception", default=0)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    #parser.add_argument("--exptime", type=int, help="exploration (no reward) time (must be < eplen)", default=0)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=3e-6)
    #parser.add_argument("--steplr", type=int, help="duration of each step in the learning rate annealing schedule", default=100000000)
    #parser.add_argument("--gamma", type=float, help="learning rate annealing factor", default=0.3)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=1000)
    parser.add_argument("--print_every", type=int, help="number of cycles between successive printing of information", default=100)
    #parser.add_argument("--", type=int, help="", default=1e-4)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    #train()
    train(argdict)

