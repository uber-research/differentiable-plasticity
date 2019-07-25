import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




##ttype = torch.FloatTensor;
#ttype = torch.cuda.FloatTensor;

#ttype = torch.FloatTensor;
#ttype = torch.cuda.FloatTensor;



class NonPlasticRNN(nn.Module):
    def __init__(self, params):
        super(NonPlasticRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs', 'fm']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 1 # For now we limit the number of neuromodulatory-output neurons to 1
        # Doesn't work with our version of PyTorch:
        #self.device = torch.device("cuda:0" if self.params['device'] == 'gpu' else "cpu")
        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).cuda()
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True) 
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()


    def forward(self, inputs, hidden): #, hebb):
        BATCHSIZE = self.params['bs']
        HS = self.params['hs']

        # Here, the *rows* of w and hebb are the inputs weights to a single neuron
        # hidden = x, hactiv = y
        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul(self.w,
                        hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        #hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
        #                hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)

        hidden = hactiv

        return activout, valueout, hidden #, hebb


    def initialZeroState(self):
        BATCHSIZE = self.params['bs']
        return Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False ).cuda()





class PlasticRNN(nn.Module):
    def __init__(self, params):
        super(PlasticRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs', 'fm']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 1 # For now we limit the number of neuromodulatory-output neurons to 1
        # Doesn't work with our version of PyTorch:
        #self.device = torch.device("cuda:0" if self.params['device'] == 'gpu' else "cpu")
        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).cuda()
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True) 
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True)
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
        #self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()

    def forward(self, inputs, hidden, hebb):
        BATCHSIZE = self.params['bs']
        HS = self.params['hs']

        # Here, the *rows* of w and hebb are the inputs weights to a single neuron
        # hidden = x, hactiv = y
        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                        hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...
        
        # deltahebb has shape BS x HS x HS
        # Each row of hebb contain the input weights to a neuron
        deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
        hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)

        hidden = hactiv

        return activout, valueout, hidden, hebb

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()

    def initialZeroState(self):
        BATCHSIZE = self.params['bs']
        return Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False ).cuda()




class SimpleModulRNN(nn.Module):
    def __init__(self, params):
        super(SimpleModulRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs', 'fm']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 1 # For now we limit the number of neuromodulatory-output neurons to 1
        # Doesn't work with our version of PyTorch:
        #self.device = torch.device("cuda:0" if self.params['device'] == 'gpu' else "cpu")
        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).cuda()
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True) 
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True)
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta (only for the non-modulated part, if any!)
        self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()

    def forward_test(self, inputs, hidden, hebb):
        NBDA = 1
        BATCHSIZE = self.params['bs']
        HS = self.params['hs']
        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul(self.w,
                        hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)
        return activout, valueout, 0, hidden, hebb

    def forward(self, inputs, hidden, hebb):
        NBDA = 1
        BATCHSIZE = self.params['bs']
        HS = self.params['hs']

        # Here, the *rows* of w and hebb are the inputs weights to a single neuron
        # hidden = x, hactiv = y
        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                        hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
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
        deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?


        hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
        if self.params['fm'] == 0:
            # Non-modulated part
            hebb2 = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
        # Soft Clamp (note that it's different from just putting a tanh on top of a freely varying value):
        #hebb1 = torch.clamp( hebb +  torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=0.0) * (1 - hebb) +  
        #        torch.clamp(DAout.view(BATCHSIZE, 1, 1)  * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
        #hebb2 = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
        # Purely additive, no clamping. This will almost certainly diverge, don't use it! 
        #hebb1 = hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
        #hebb2 = hebb + self.eta * deltahebb

        if self.params['fm'] == 1:
            hebb = hebb1
        elif self.params['fm'] == 0:
            # Combine the modulated and non-modulated part
            hebb = torch.cat( (hebb1[:, :self.params['hs']//2, :], hebb2[:,  self.params['hs'] // 2:, :]), dim=1) # Maybe along dim=2 instead?...
        else:
            raise ValueError("Must select whether fully modulated or not (params['fm'])")

        hidden = hactiv

        return activout, valueout, DAout, hidden, hebb

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()

    def initialZeroState(self):
        BATCHSIZE = self.params['bs']
        return Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False ).cuda()





class RetroModulRNN(nn.Module):
    def __init__(self, params):
        super(RetroModulRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs', 'fm']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 1 # For now we limit the number of neuromodulatory-output neurons to 1
        # Doesn't work with our version of PyTorch:
        #self.device = torch.device("cuda:0" if self.params['device'] == 'gpu' else "cpu")
        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).cuda()
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True) 
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).cuda(), requires_grad=True)
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta (only for the non-modulated part, if any!)
        self.etaet = torch.nn.Parameter((.1 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same etaet
        self.h2DA = torch.nn.Linear(params['hs'], NBDA).cuda()
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).cuda()
        self.h2v = torch.nn.Linear(params['hs'], 1).cuda()

    def forward(self, inputs, hidden, hebb, et, pw):
            NBDA = 1
            BATCHSIZE = self.params['bs']
            HS = self.params['hs']
    
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, pw)),
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
            
            if self.params['rule'] == 'hebb':
                deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?
            elif self.params['rule'] == 'oja':
                deltahebb =  torch.mul(hactiv.view(BATCHSIZE, HS, 1), (hidden.view(BATCHSIZE, 1, HS) - torch.mul(self.w.view(1, HS, HS), hactiv.view(BATCHSIZE, HS, 1))))
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")

            # Hard clamp
            deltapw = DAout.view(BATCHSIZE,1,1) * et
            pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
            
            # Should we have a fully neuromodulated network, or only half?
            if self.params['fm'] == 1:
                pw = pw1
            elif self.params['fm']==0:
                hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
                pw = torch.cat( (hebb[:, :self.params['hs']//2, :], pw1[:,  self.params['hs'] // 2:, :]), dim=1) # Maybe along dim=2 instead?...
            else:
                raise ValueError("Must select whether fully modulated or not")

            # Updating the eligibility trace - always a simple decay term. 
            # Note that self.etaet != self.eta (which is used for hebb, i.e. the non-modulated part)
            deltaet = deltahebb
            et = (1 - self.etaet) * et + self.etaet *  deltaet
            
            hidden = hactiv
            return activout, valueout, DAout, hidden, hebb, et, pw
        
        
        

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()
    
    def initialZeroPlasticWeights(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']) , requires_grad=False).cuda()
    def initialZeroState(self):
        return Variable(torch.zeros(self.params['bs'], self.params['hs']), requires_grad=False ).cuda()















